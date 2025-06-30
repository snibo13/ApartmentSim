# import gemini_stream
from gemini_stream import BlenderRenderer
import superpoint_test
import matplotlib.pyplot as plt
import cv2
from PIL import Image

blend_path = "apartment.blend"
# Initialize the renderer with the blend file and camera
renderer = BlenderRenderer(blend_filepath=blend_path, camera_name="robo_cam")
bot = superpoint_test.Bot()

frames = []
rs = []
ts = []
# Example usage: Stream and render frames
frame = renderer.render_frame_to_numpy()
frame_prev = frame

accumulated_poses = []
accumulated_poses.append(bot.current_pose[:3, 3].copy()) # Store current pose after update

movements = [
    {'dx': 1.0, 'dy': 0.0, 'dz': 0.0},
    {'dx': 0.0, 'dy': 1.0, 'dz': 0.0},
    {'dx': -1.0, 'dy': 0.0, 'dz': 0.0},
    {'dx': 0.0, 'dy': -1.0, 'dz': 0.0}
]

for i, move in enumerate(movements):
    print(f"\n--- Processing Movement {i+1}: dx={move['dx']}, dy={move['dy']}, dz={move['dz']} ---")
    
    # Move camera in Blender
    renderer.move_camera(dx=move['dx'], dy=move['dy'], dz=move['dz'])
    
    # Render new frame
    frame_curr = renderer.render_frame_to_numpy()
    plt.imshow("Current frame", Image.fromarray(frame_curr))
    cv2.waitKey(20)
    if frame_curr is None:
        print(f"Failed to render frame after movement {i+1}. Skipping.")
        frame_prev = None # Invalidate previous frame to prevent issues
        continue # Skip this iteration

    outputs = superpoint_test.run_super_glue([frame_prev, frame_curr])
    
    if outputs and len(outputs["keypoints0"]) > 0:
        print(f"Matches found: {len(outputs['keypoints0'])}")
        R,t = superpoint_test.compute_essential_and_scale(outputs, bot)
    
    else:
        print("No matches or matching failed, skipping pose estimation for this step.")

    accumulated_poses.append(bot.current_pose[:3, 3].copy()) # Store current pose after update

    # Prepare for next iteration
    frame_prev = frame_curr

# --- Plotting the Trajectory ---
# Extract X, Y, Z coordinates from the accumulated poses
x_coords = [pose[0] for pose in accumulated_poses]
y_coords = [pose[1] for pose in accumulated_poses]
z_coords = [pose[2] for pose in accumulated_poses]

plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', label='Estimated Trajectory (XY)')
# You might want to flip the y-axis if Blender's Y-up differs from your plot convention.
plt.gca().invert_yaxis() # Uncomment if Y-axis needs inversion

# Plot starting point
plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
# Plot ending point
plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')

plt.title('Estimated Camera Trajectory (XY Plane)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.axis('equal') # Equal aspect ratio to prevent distortion
plt.legend()
plt.show()

# Optional: Plot 3D trajectory
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_coords, y_coords, z_coords, marker='o', linestyle='-', color='blue', label='Estimated Trajectory (3D)')
# ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='g', s=100, label='Start')
# ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='r', s=100, label='End')
# ax.set_title('Estimated Camera Trajectory (3D)')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.legend()
# plt.show()

print("\nScript finished.")