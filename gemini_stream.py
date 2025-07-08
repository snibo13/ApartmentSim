import bpy
import cv2
import os
import numpy as np
import sys
import tempfile
import cProfile


class BlenderRenderer:
    """
    A class to handle Blender rendering operations, including scene setup,
    frame rendering, and headless streaming.
    """

    def __init__(self, blend_filepath, camera_names=["robo_cam"]):
        """
        Initializes the BlenderRenderer.

        Args:
            blend_filepath (str): The path to the Blender file (.blend).
            camera_name (str): The name of the camera object to use for rendering.
        """
        self.blend_filepath = blend_filepath
        self.camera_names = camera_names
        self.current_rendered_frame = (
            None  # To store the last rendered frame as a NumPy array
        )
        if blend_filepath.endswith(".usdc"):
            self._load_usdc_file()
        else:
            self._load_blend_file()
        self._setup_render_settings()

    def _load_usdc_file(self):
        """
        Loads the specified USD file into Blender.
        """
        if not os.path.exists(self.blend_filepath):
            print(f"ERROR: USD file not found: {self.blend_filepath}")
            sys.exit(1)

        print(f"Loading .usdc file: {self.blend_filepath}")
        bpy.ops.wm.usd_import(filepath=self.blend_filepath)
        self.scene = bpy.context.scene

    def _load_blend_file(self):
        """
        Loads the specified Blender file.
        """
        if not os.path.exists(self.blend_filepath):
            print(f"ERROR: Blend file not found: {self.blend_filepath}")
            sys.exit(1)

        print(f"Loading .blend file: {self.blend_filepath}")
        bpy.ops.wm.open_mainfile(filepath=self.blend_filepath)
        self.scene = bpy.context.scene

    def _setup_render_settings(self):
        """
        Sets up the render engine, camera, and output format for the scene.
        """
        self.scene.render.engine = "BLENDER_EEVEE_NEXT"
        # self.scene.eevee.samples = 1 # Eevee Next does not have samples in the same way
        self.scene.eevee.taa_render_samples = 25
        # self.scene.eevee.taa_samples = 5
        # self.scene.eevee.use_gtao = True
        bpy.context.scene.render.resolution_percentage = 100
        # bpy.context.scene.render.use_simplify = True

        for camera_name in self.camera_names:
            print(f"Checking for camera: {camera_name}")
            if camera_name in bpy.data.objects:
                print(f"Found camera: {camera_name}")
                self.scene.camera = bpy.data.objects[camera_name]
            else:
                print(f"ERROR: Camera '{camera_name}' not found in the scene.")
                sys.exit(1)

        self.scene.render.image_settings.file_format = "PNG"
        # The specific filepath will be set per frame in render_frame_to_file or stream_headless

    def get_camera_intrinsics_blender(self, camera_object=None):
        """
        Calculates the camera intrinsic matrix (K) from Blender's camera parameters.

        Args:
            camera_object (bpy.types.Object, optional): The camera object to get intrinsics from.
                                                         If None, uses the active scene camera.

        Returns:
            np.ndarray: A 3x3 NumPy array representing the intrinsic camera matrix K.
        """
        if camera_object is None:
            camera_object = self.scene.camera
            if not camera_object:
                print(
                    "Error: No camera object specified and no active scene camera found."
                )
                return None
            if camera_object.type != "CAMERA":
                print(f"Error: Object '{camera_object.name}' is not a camera.")
                return None

        # Get camera data (bpy.types.Camera) from the camera object
        cam_data = camera_object.data

        # Get render resolution
        render_props = self.scene.render
        resolution_x = render_props.resolution_x
        resolution_y = render_props.resolution_y
        scale = render_props.resolution_percentage / 100

        # Effective resolution after scaling
        render_width = resolution_x * scale
        render_height = resolution_y * scale

        # Get sensor width and focal length
        # Blender's focal length is `lens` in millimeters
        f_in_mm = cam_data.lens  # Focal length in millimeters

        # Sensor size in millimeters
        # Blender's `sensor_fit` determines how `sensor_width` and `sensor_height` are used.
        # For 'AUTO' or 'HORIZONTAL', `sensor_width` is usually the dominant one.
        # We need to account for the sensor aspect ratio if it's not square.
        sensor_width_in_mm = cam_data.sensor_width
        sensor_height_in_mm = cam_data.sensor_height

        # Calculate focal length in pixels (fx, fy)
        # fx = f_in_mm * (pixels_per_mm_x)
        # fy = f_in_mm * (pixels_per_mm_y)

        # pixels_per_mm_x = render_width / sensor_width_in_mm
        # pixels_per_mm_y = render_height / sensor_height_in_mm (if sensor_fit is VERTICAL)
        # Or derived from sensor_width and aspect ratio if sensor_fit is HORIZONTAL/AUTO

        # This calculation needs to correctly handle `sensor_fit`
        # and pixel aspect ratio.
        print(f"Render resolution: {render_width}x{render_height}")
        print(f"Sensor size: {sensor_width_in_mm}mm x {sensor_height_in_mm}mm")
        pixel_aspect_ratio = render_props.pixel_aspect_x / render_props.pixel_aspect_y
        print(f"Pixel aspect ratio: {pixel_aspect_ratio}")
        print(
            f"Pixel aspect ratio: {render_props.pixel_aspect_x} / {render_props.pixel_aspect_y}"
        )

        # Determine the effective sensor size based on `sensor_fit`
        if cam_data.sensor_fit == "VERTICAL":
            # Sensor fit to vertical, so height is fixed.
            # Width scales with aspect ratio.
            s_u = render_width / (sensor_height_in_mm * pixel_aspect_ratio)
            s_v = render_height / sensor_height_in_mm
        else:  # 'HORIZONTAL' or 'AUTO' (which typically behaves like HORIZONTAL for landscape renders)
            # Sensor fit to horizontal, so width is fixed.
            # Height scales with aspect ratio.
            s_u = render_width / sensor_width_in_mm
            s_v = render_height / (sensor_width_in_mm / pixel_aspect_ratio)
            print(f"Render height: {render_height}, Sensor width: {sensor_width_in_mm}")
            print(
                f"Using horizontal sensor fit: s_u={s_u}, s_v={s_v} (aspect ratio {pixel_aspect_ratio})"
            )

        fx = f_in_mm * s_u
        fy = f_in_mm * s_v

        # Principal point (cx, cy)
        # Default is center of the image. Blender's shift_x/y move the principal point.
        # shift_x/y are normalized to the sensor size, so they need to be scaled by resolution.
        # Blender's Y axis is typically "up", while image coordinates have Y "down".
        # cx = (render_width / 2) - (cam_data.shift_x * render_width)
        # cy = (render_height / 2) + (cam_data.shift_y * render_height) # Blender's shift_y positive is UP, CV is DOWN

        # A more common and robust way for principal point for CV:
        # cx and cy are the pixel coordinates of the principal point.
        # For a centered camera with no shift, cx = width / 2, cy = height / 2.
        # shift_x and shift_y directly relate to the offset of the principal point.
        # Blender's `shift_x` and `shift_y` are relative to the *sensor dimensions*.
        # They express the shift as a fraction of the sensor width/height.

        # When sensor_fit is HORIZONTAL or AUTO:
        cx = render_width / 2.0 - cam_data.shift_x * s_u
        cy = (
            render_height / 2.0 + cam_data.shift_y * s_v
        )  # Y-axis inversion for CV convention

        print(f"Focal lengths: fx={fx}, fy={fy}")
        print(f"Principal point: cx={cx}, cy={cy}")

        # If sensor_fit is VERTICAL, then shift_x and shift_y are relative to sensor_height
        # and render_height, and need to be scaled by aspect ratio.
        if cam_data.sensor_fit == "VERTICAL":
            cx = render_width / 2.0 - cam_data.shift_x * s_u * pixel_aspect_ratio
            cy = render_height / 2.0 + cam_data.shift_y * s_v

        # Construct the intrinsic matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        return K

    def render_frame_to_file(self, filepath):
        """
        Renders the current frame and saves it to the specified filepath.

        Args:
            filepath (str): The full path to save the rendered image.
        """
        print(f"Rendering frame to: {filepath}")
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        print(f"Finished rendering: {filepath}")
        # Update current_rendered_frame if it's the 'current_frame.png'
        if "current_frame.png" in filepath:
            self.read_current_frame_from_file()

    def render_frame_to_numpy(self):
        """
        Renders the current frame to a temporary file, loads it into a NumPy array,
        and then deletes the temporary file. This is the most efficient method
        for headless mode to get pixel data.
        """
        # Create a temporary file path
        # Using NamedTemporaryFile ensures a unique filename and handles deletion
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_filepath = tmp_file.name

        self.scene.render.filepath = temp_filepath

        # In headless mode, write_still=True is generally required for the render result
        # to be saved/available, even if it's just to a temporary path.
        bpy.ops.render.render(write_still=True)

        # Load the rendered image from the temporary file
        img = cv2.imread(temp_filepath, cv2.IMREAD_COLOR)

        # Clean up the temporary file immediately
        try:
            os.remove(temp_filepath)
        except OSError as e:
            print(f"Error removing temporary file {temp_filepath}: {e}")

        if img is not None:
            self.current_rendered_frame = img
            return img
        else:
            print(f"ERROR: Could not read image from temporary file {temp_filepath}")
            self.current_rendered_frame = None
            return None

    def render_frames_to_numpy(self):
        imgs = []
        for camera_name in self.camera_names:
            print(f"Setting active camera: {camera_name}")
            self.scene.camera = bpy.data.objects[camera_name]
            imgs.append(self.render_frame_to_numpy())
        return imgs

    def read_current_frame_from_file(
        self, filepath="rendered_frames/current_frame.png"
    ):
        """
        Reads the 'current_frame.png' file into the current_rendered_frame attribute
        and displays it.
        """
        if os.path.exists(filepath):
            self.current_rendered_frame = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if self.current_rendered_frame is None:
                print(f"ERROR: Could not read image from {filepath}")
        else:
            print(f"ERROR: File not found: {filepath}")

    def render_single_frame_headless(
        self, frame_number=1, output_dir="rendered_frames"
    ):
        """
        Renders a single frame in headless mode.

        Args:
            frame_number (int): The frame number to render.
            output_dir (str): The directory to save the rendered frame.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"Starting headless frame rendering for frame {frame_number}")

        # Set the frame to render
        self.scene.frame_set(frame_number)

        # Construct filepath for the current frame
        frame_filename = f"frame_{frame_number:04d}.png"  # e.g., frame_0001.png
        frame_filepath = os.path.join(output_dir, frame_filename)

        # Render the frame
        self.render_frame_to_file(frame_filepath)

        # Also render and update the 'current_frame.png' for immediate access
        current_frame_path = os.path.join(output_dir, "current_frame.png")
        self.render_frame_to_file(current_frame_path)
        self.read_current_frame_from_file(current_frame_path)

        print(f"Finished rendering frame {frame_number}")

    def stream_headless(
        self, start_frame=1, end_frame=50, output_dir="rendered_frames"
    ):
        """
        Renders a sequence of frames in headless mode and saves them to files.
        Optionally updates an in-memory representation of the "current" frame.

        Args:
            start_frame (int): The starting frame number.
            end_frame (int): The ending frame number.
            output_dir (str): The directory to save the rendered frames.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"Starting headless frame rendering... ({start_frame} to {end_frame})")

        for frame in range(start_frame, end_frame + 1):
            self.scene.frame_set(frame)
            # Construct filepath for the current frame
            frame_filename = f"frame_{frame:04d}.png"  # e.g., frame_0001.png
            frame_filepath = os.path.join(output_dir, frame_filename)
            self.render_frame_to_file(frame_filepath)

            # Render and update the 'current_frame.png' for immediate access
            self.render_frame_to_file(os.path.join(output_dir, "current_frame.png"))
            self.read_current_frame_from_file(
                os.path.join(output_dir, "current_frame.png")
            )

            # If you wanted to get the numpy array of each rendered frame directly:
            # self.render_frame_to_numpy() # This will update self.current_rendered_frame with each frame

        print(
            f"Finished rendering {end_frame - start_frame + 1} frames to '{output_dir}'."
        )

    def move_camera(self, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
        """
        Moves and rotates the active camera.

        Args:
            dx (float): Change in X position.
            dy (float): Change in Y position.
            dz (float): Change in Z position.
            rx (float): Rotation around X axis (in degrees).
            ry (float): Rotation around Y axis (in degrees).
            rz (float): Rotation around Z axis (in degrees).
        """
        for camera_name in self.camera_names:
            camera = bpy.data.objects[camera_name]
            if camera:
                # Apply location changes
                camera.location.x += dx
                camera.location.y += dy
                camera.location.z += dz

                # Apply rotation changes (convert degrees to radians for Blender)
                camera.rotation_euler.x += np.radians(rx)
                camera.rotation_euler.y += np.radians(ry)
                camera.rotation_euler.z += np.radians(rz)
                print(
                    f"Camera moved to: {camera.location}, Rotated to: {np.degrees(camera.rotation_euler.x):.2f}, {np.degrees(camera.rotation_euler.y):.2f}, {np.degrees(camera.rotation_euler.z):.2f} degrees"
                )
            else:
                print("No camera available to move.")


def main():
    blend_path = "apartment.blend"  # Make sure this path is correct

    # Initialize the renderer with the blend file and camera
    renderer = BlenderRenderer(blend_filepath=blend_path, camera_names="robo_cam")

    # Example usage: Stream and render frames
    renderer.stream_headless(start_frame=1, end_frame=3)  # Render frames 1 to 10

    print(renderer.get_camera_intrinsics_blender())

    print("Script finished.")


if __name__ == "__main__":
    # This block ensures the script runs correctly whether executed
    # normally or from within Blender's Python environment.
    try:
        # Check if bpy is already available (running inside Blender)
        if "bpy" in sys.modules:
            cProfile.run("main()")
        else:
            print("ERROR: This script must be run within Blender's Python environment.")
            print(
                "Please run Blender and execute this script from File -> Execute Script,"
            )
            print("or by using 'blender -b -P your_script_name.py'")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
