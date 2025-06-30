import bpy
import cv2
import os
import numpy as np
import sys

# Ensure this script is run with Blender's Python environment
# If running outside Blender, these imports will fail.


def setup_render_settings(scene, camera_name="robo_cam"):
    """
    Sets up the render engine, camera, and output format.
    """
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    # scene.eevee.samples = 1

    if camera_name in bpy.data.objects:
        scene.camera = bpy.data.objects[camera_name]
    else:
        print(
            f"WARNING: Camera '{camera_name}' not found. Using active camera if available."
        )
        if not scene.camera:
            print(
                "ERROR: No camera found in scene. Please add a camera or specify a valid camera name."
            )
            sys.exit(1)

    scene.render.image_settings.file_format = "PNG"
    # We will set the specific filepath per frame in the stream_headless function


def render_frame_to_numpy(scene):
    """
    Renders the current frame and returns it as a NumPy array.
    """
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Get the rendered image
    img = bpy.data.images["Render Result"]
    if not img:
        print("ERROR: Render Result image not found.")
        return None
    if img.is_dirty:
        img.update()
    print("Rendering complete. Converting image to NumPy array...")
    # Ensure the image is loaded
    if not img.has_data:
        print("ERROR: Render Result image has no data.")
        return None

    # Convert to NumPy array
    pixels = np.array(img.pixels[:])
    width, height = img.size
    image = (pixels * 255).astype(np.uint8).reshape((height, width, 4))

    # Convert from RGBA to BGR for OpenCV compatibility
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.imshow("Rendered Frame", image)
    cv2.waitKey(1)  # Allow OpenCV to update the window

    return image


def render_frame_to_file(scene, filepath):
    """
    Renders the current frame and saves it to the specified filepath.
    """
    print(f"Rendering frame to: {filepath}")
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"Finished rendering: {filepath}")


def stream_headless(start=1, end=50, output_dir="rendered_frames"):
    """
    Renders a sequence of frames in headless mode and saves them to files.
    """
    scene = bpy.context.scene

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Starting headless frame rendering... ({start} to {end})")

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        # Construct filepath for the current frame
        frame_filename = f"frame_{frame:04d}.png"  # e.g., frame_0001.png
        frame_filepath = os.path.join(output_dir, frame_filename)
        # render_frame_to_file(scene, frame_filepath)
        render_frame_to_numpy(scene)

    print(f"Finished rendering {end - start + 1} frames to '{output_dir}'.")


def main():
    blend_path = "apartment.blend"

    if not os.path.exists(blend_path):
        print(f"ERROR: Blend file not found: {blend_path}")
        sys.exit(1)

    print(f"Loading .blend file: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Set up render settings after loading the blend file
    setup_render_settings(bpy.context.scene, camera_name="robo_cam")

    # Call stream_headless to render and save frames
    stream_headless(start=1, end=100)  # Render frames 1 to 100

    print("Script finished.")


if __name__ == "__main__":
    main()
