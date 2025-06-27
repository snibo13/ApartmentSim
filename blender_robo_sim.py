import bpy
import os

# --- Configuration ---
# Path to your Blender file (replace with your .blend file)
# If you run this script directly in Blender with a scene open,
# you might not need to explicitly load a file if you're operating on the current scene.
# For headless execution (blender --background your_scene.blend --python script.py),
# your_scene.blend will be loaded first.
BLENDER_FILE_PATH = "apartment.blend" # e.g., "/Users/youruser/Documents/my_robot_scene.blend"

# Output directory for rendered images
OUTPUT_DIR = "rendered_views"

# Scene to render (usually 'Scene' or the name of your scene)
SCENE_NAME = "Scene" # Replace if your scene has a different name

# Camera name in your Blender scene
CAMERA_NAME = "robo_cam" # Replace if your camera has a different name

# --- Setup Functions ---

def ensure_output_directory(output_path):
    """Ensures the output directory exists."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

def load_blender_scene(file_path):
    """
    Loads a Blender scene from a .blend file.
    Note: This will replace the current scene in Blender.
    For typical robotics simulation, you'll likely have a base .blend file
    with your robot and environment.
    """
    if not os.path.exists(file_path):
        print(f"Error: Blender file not found at {file_path}")
        return False
    
    try:
        bpy.ops.wm.open_mainfile(filepath=file_path)
        print(f"Successfully loaded scene: {file_path}")
        return True
    except Exception as e:
        print(f"Failed to load scene: {e}")
        return False

def get_target_scene(scene_name):
    """Returns the bpy.data.scenes object for the specified scene name."""
    if scene_name not in bpy.data.scenes:
        print(f"Error: Scene '{scene_name}' not found.")
        return None
    return bpy.data.scenes[scene_name]

def get_target_camera(scene, camera_name):
    """Returns the bpy.data.objects object for the specified camera name."""
    if camera_name not in bpy.data.objects:
        print(f"Error: Camera '{camera_name}' not found.")
        return None
    
    camera_obj = bpy.data.objects[camera_name]
    if camera_obj.type == 'CAMERA':
        scene.camera = camera_obj # Set this camera as the active scene camera
        print(f"Successfully set '{camera_name}' as active camera.")
        return camera_obj
    else:
        print(f"Error: Object '{camera_name}' is not a camera.")
        return None

def set_camera_transform(camera_obj, location=None, rotation_euler=None):
    """
    Sets the location and/or rotation (Euler angles in radians) of the camera.
    
    Args:
        camera_obj (bpy.types.Object): The camera object to transform.
        location (tuple): (x, y, z) coordinates. None to keep current.
        rotation_euler (tuple): (roll, pitch, yaw) in radians. None to keep current.
    """
    if location:
        camera_obj.location = location
        print(f"Camera '{camera_obj.name}' moved to: {location}")
    if rotation_euler:
        camera_obj.rotation_mode = 'XYZ' # Ensure Euler mode is XYZ
        camera_obj.rotation_euler = rotation_euler
        print(f"Camera '{camera_obj.name}' rotated to: {rotation_euler} (Euler radians)")

def render_scene(scene, output_path_prefix, resolution_x=1920, resolution_y=1080):
    """
    Renders the scene from the active camera and saves the image.
    
    Args:
        scene (bpy.types.Scene): The scene to render.
        output_path_prefix (str): Prefix for the output image file (e.g., "view_1").
        resolution_x (int): Horizontal resolution of the render.
        resolution_y (int): Vertical resolution of the render.
    """
    ensure_output_directory(OUTPUT_DIR) # Make sure the base output dir exists

    # Set render settings
    scene.render.engine = 'BLENDER_EEVEE' # Or 'BLENDER_EEVEE' for faster renders
    scene.render.image_settings.file_format = 'PNG' # Or 'JPEG', 'OPEN_EXR', etc.
    scene.render.filepath = os.path.join(OUTPUT_DIR, f"{output_path_prefix}.png")
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.film_transparent = True # Set to False if you want background visible

    print(f"Rendering scene to: {scene.render.filepath}")
    
    try:
        bpy.ops.render.render(write_still=True)
        print(f"Render completed for {output_path_prefix}.png")
    except Exception as e:
        print(f"Error during rendering: {e}")

# --- Main Robotics Simulation Loop ---

def main():
    """Main function to orchestrate loading, moving, and rendering."""
    print("\n--- Starting Blender Robotics Simulation Script ---")

    # 1. Load the Blender Scene (only if running as a standalone script or fresh start)
    # If you run this directly in an open Blender instance, comment out the load_blender_scene
    # and ensure your desired scene/camera are already active.
    # if not load_blender_scene(BLENDER_FILE_PATH):
    #     print("Exiting due to scene loading error.")
    #     return

    scene = get_target_scene(SCENE_NAME)
    if not scene:
        return

    camera_obj = get_target_camera(scene, CAMERA_NAME)
    if not camera_obj:
        return

    # Example 1: Initial Render from default camera position
    print("\n--- Performing initial render ---")
    render_scene(scene, "initial_view")

    # Example 2: Move camera to a new position and render
    print("\n--- Moving camera to position 1 and re-rendering ---")
    # Define new camera position (X, Y, Z) and rotation (roll, pitch, yaw in radians)
    # Blender's default camera looks along -Z.
    # To look at the origin from +Y, camera is at (0, -some_dist, 0) and rotated X=90deg (pi/2)
    # math.radians() is useful for converting degrees to radians.
    import math
    set_camera_transform(
        camera_obj,
        location=(0, -5.0, 2.0), # Example: 5 units back, 2 units up
        rotation_euler=(math.radians(70), math.radians(0), math.radians(0)) # Look down slightly
    )
    render_scene(scene, "view_position_1")

    # Example 3: Move camera to another position and render
    print("\n--- Moving camera to position 2 and re-rendering ---")
    set_camera_transform(
        camera_obj,
        location=(3.0, 3.0, 1.5), # Example: Diagonal view
        rotation_euler=(math.radians(60), math.radians(0), math.radians(135)) # Look towards origin
    )
    render_scene(scene, "view_position_2")

    # Example 4: Iterate through a series of camera poses (e.g., a sweep or path)
    print("\n--- Iterating through a series of camera poses ---")
    camera_poses = [
        {"location": (-4.0, 0.0, 1.0), "rotation_euler": (math.radians(90), 0, math.radians(-90))}, # Side view
        {"location": (0.0, 4.0, 1.0), "rotation_euler": (math.radians(90), 0, math.radians(180))},  # Front view
        {"location": (0.0, -4.0, 1.0), "rotation_euler": (math.radians(90), 0, math.radians(0))}   # Back view
    ]

    for i, pose in enumerate(camera_poses):
        print(f"\n--- Moving to pose {i+1} ---")
        set_camera_transform(camera_obj, pose["location"], pose["rotation_euler"])
        render_scene(scene, f"view_pose_{i+1}")

    print("\n--- Blender Robotics Simulation Script Finished ---")

# --- Execute the main function ---
if __name__ == "__main__":
    main()

