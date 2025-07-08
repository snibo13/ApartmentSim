import os
import cv2
import numpy as np
import bpy
import matplotlib.pyplot as plt
from gemini_stream import BlenderRenderer
from superpoint_test import run_super_glue


# --- Utility to extract Blender camera pose as a 4x4 matrix ---
def get_camera_pose_matrix(camera):
    mat = np.array(camera.matrix_world)
    return mat


# --- Set up paths and renderer with stereo cameras ---
blend_path = os.path.join(os.path.dirname(__file__), "apartment.blend")
renderer = BlenderRenderer(
    blend_filepath=blend_path, camera_names=["robo_cam_l", "robo_cam_r"]
)

# --- Get intrinsics and stereo baseline ---
K = renderer.get_camera_intrinsics_blender(bpy.data.objects["robo_cam_l"])

cam_l = bpy.data.objects["robo_cam_l"]
cam_r = bpy.data.objects["robo_cam_r"]
T_world_l = get_camera_pose_matrix(cam_l)
T_world_r = get_camera_pose_matrix(cam_r)
T_rl = np.linalg.inv(T_world_l) @ T_world_r  # Right-to-Left transform

# Add after T_rl calculation
baseline = np.linalg.norm(T_rl[:3, 3])
print(f"Stereo baseline: {baseline:.3f} units")
print("T_rl translation:", T_rl[:3, 3])

# --- Render stereo pair ---
stereo_frames = renderer.render_frames_to_numpy()
frame_left = stereo_frames[0]
frame_right = stereo_frames[1] if len(stereo_frames) > 1 else None

depth_img = cv2.imread(
    os.path.join(os.path.dirname(__file__), "depth.png/Image0000.png"),
    cv2.IMREAD_UNCHANGED,
)


# --- Run SuperGlue to match keypoints ---
if frame_left is not None and frame_right is not None:
    outputs = run_super_glue([frame_left, frame_right], visualise=True)

    # --- Triangulate 3D points from stereo ---
    if outputs and len(outputs["keypoints0"]) >= 8:
        pts_left = outputs["keypoints0"].numpy().T  # shape (2, N)
        pts_right = outputs["keypoints1"].numpy().T

        # Projection matrices
        P0 = K @ np.eye(3, 4)
        P1 = K @ T_rl[:3, :]

        pts_left = outputs["keypoints0"].numpy().astype(np.float64).T  # (2, N)
        pts_right = outputs["keypoints1"].numpy().astype(np.float64).T  # (2, N)

        # Show the images side by side with the keypoints linked
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(frame_left)
        # plt.scatter(pts_left[0], pts_left[1], s=5, c="red", label="Left Keypoints")
        # plt.title("Left Camera Keypoints")
        # plt.axis("off")
        # plt.subplot(1, 2, 2)
        # plt.imshow(frame_right)
        # plt.scatter(pts_right[0], pts_right[1], s=5, c="blue", label="Right Keypoints")
        # plt.title("Right Camera Keypoints")
        # plt.axis("off")
        # plt.legend()
        # plt.show()

        # input()

        K = K.astype(np.float64)
        T_rl = T_rl.astype(np.float64)

        # Build projection matrices explicitly
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        R_rl = T_rl[:3, :3]
        t_rl = T_rl[:3, 3].reshape(3, 1)
        P1 = K @ np.hstack((R_rl, t_rl))

        # Make sure keypoints are in pixel coordinates (2,N float64)
        # If keypoints normalized, multiply by image width and height here

        pts_left = outputs["keypoints0"].numpy().astype(np.float64).T
        pts_right = outputs["keypoints1"].numpy().astype(np.float64).T

        pts_4d_hom = cv2.triangulatePoints(P0, P1, pts_left, pts_right)
        pts_3d = (pts_4d_hom[:3] / pts_4d_hom[3]).T

        depth = -pts_3d[:, 2]  # Confirm depth sign is correct

        print("Sample depths:", depth[:10])
        print("Baseline:", baseline)
        # Normalize the depth for visualization

        # Project the 3D points back to the left image
        pts_2d_proj = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, None)[
            0
        ].squeeze()
        pts_2d_proj = pts_2d_proj.astype(np.float32)
        pts_2d_proj = pts_2d_proj.reshape(-1, 2)
        # Draw the projected points on the left image
        img_to_draw = depth_img.copy()
        for pt, depth_val in zip(pts_2d_proj, depth):
            color = (0, int(255 * depth_val), int(255 * (1 - depth_val)))
            # Label the points with their depth
            cv2.putText(
                img_to_draw,
                f"{depth_val:.2f}",
                tuple(pt.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            cv2.circle(img_to_draw, tuple(pt.astype(int)), 3, color, -1)
        plt.imshow(cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB))
        plt.title("Left Camera with Projected 3D Points")
        plt.axis("off")
        plt.show()

    #     # Filter invalid points
    #     valid = (
    #         np.isfinite(pts_3d).all(axis=1)
    #         & (pts_3d[:, 2] > 0)
    #         & (np.linalg.norm(pts_3d, axis=1) < 10)
    #     )
    #     pts_3d = pts_3d[valid]
    #     kp0_valid = outputs["keypoints0"].numpy()[valid]

    #     # --- Simulate next motion step ---
    #     renderer.move_camera(dx=0.5)  # Move forward in X
    #     new_frame_left = renderer.render_frame_to_numpy()

    #     # --- Track features from previous left frame to new left frame ---
    #     track_outputs = run_super_glue([frame_left, new_frame_left], visualise=True)

    #     if track_outputs and len(track_outputs["keypoints0"]) >= 8:
    #         match_indices_0 = track_outputs["match_indices0"].numpy()
    #         match_indices_1 = track_outputs["match_indices1"].numpy()
    #         kp_prev = track_outputs["keypoints0"].numpy()
    #         kp_curr = track_outputs["keypoints1"].numpy()

    #         matched_3d = []
    #         matched_2d = []
    #         for i, j in enumerate(match_indices_0):
    #             if j != -1:
    #                 ref_kp = kp_prev[i]
    #                 for k, ref in enumerate(kp0_valid):
    #                     if np.linalg.norm(ref_kp - ref) < 1.0:
    #                         matched_3d.append(pts_3d[k])
    #                         matched_2d.append(kp_curr[j])

    #         if len(matched_3d) >= 6:
    #             matched_3d = np.array(matched_3d).reshape(-1, 3).astype(np.float32)
    #             matched_2d = np.array(matched_2d).reshape(-1, 2).astype(np.float32)

    #             _, rvec, tvec, inliers = cv2.solvePnPRansac(
    #                 matched_3d, matched_2d, K, None
    #             )

    #             # Convert estimated pose to a 4x4 matrix
    #             R, _ = cv2.Rodrigues(rvec)
    #             T_est = np.eye(4)
    #             T_est[:3, :3] = R
    #             T_est[:3, 3] = tvec.flatten()

    #             # Get Blender ground truth pose after camera move
    #             T_gt = get_camera_pose_matrix(bpy.data.objects["robo_cam_l"])

    #             # Compute relative pose error
    #             T_error = np.linalg.inv(T_gt) @ T_est
    #             t_error = T_error[:3, 3]
    #             angle_error = np.arccos((np.trace(T_error[:3, :3]) - 1) / 2)
    #             angle_error_deg = np.degrees(angle_error)

    #             print("\nEstimated pose from stereo tracking:")
    #             print("Rotation vector (deg):", np.degrees(rvec).flatten())
    #             print("Translation vector:", tvec.flatten())
    #             print("\nGround truth translation:", T_gt[:3, 3])
    #             print("Pose error (translation, meters):", np.linalg.norm(t_error))
    #             print("Pose error (rotation, degrees):", angle_error_deg)
    #         else:
    #             print("Not enough 2D-3D matches for pose estimation.")
    #     else:
    #         print("Feature tracking failed.")

    #     # --- Plot 3D points ---
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], s=2, c="blue")
    #     ax.set_title("Triangulated 3D Points (Left Camera Frame)")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     plt.show()
    # else:
    #     print("Not enough matches to triangulate.")
else:
    print("Failed to render both stereo frames.")

# Add debug print to check your camera parameters
print("Camera intrinsics K:")
print(K)
print("Image dimensions:", frame_left.shape[:2])
