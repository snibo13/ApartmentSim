from transformers import AutoImageProcessor, SuperPointForKeypointDetection #SuperPoint
from transformers import AutoImageProcessor, AutoModel # SuperGlue
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_indoor")
model = AutoModel.from_pretrained("magic-leap-community/superglue_indoor")

def run_super_glue(images, visualise=False):
    inputs = processor(images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    image1 = Image.fromarray(images[0])
    image2 = Image.fromarray(images[1])
    images = [image1, image2]

    merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
    merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
    merged_image[: image2.height, image1.width :] = np.array(image2) / 255.0
    # plt.imshow(merged_image)
    # plt.axis("off")

    # Retrieve the keypoints and matches
    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    output = outputs[0]
    keypoints0 = output["keypoints0"]
    keypoints1 = output["keypoints1"]
    matching_scores = output["matching_scores"]
    keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
    keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

    if visualise:
        # Plot the matches
        for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
                keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
        ):
            plt.plot(
                [keypoint0_x, keypoint1_x + image1.width],
                [keypoint0_y, keypoint1_y],
                color=plt.get_cmap("RdYlGn")(matching_score.item()),
                alpha=0.9,
                linewidth=0.5,
            )
            plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
            plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)

        # Save the plot
        plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
        plt.close()
    return output

def compute_essential_and_scale(output, bot):
    matched_kp0 = output["keypoints0"].numpy().astype(np.float32).reshape(-1, 1, 2)
    matched_kp1 = output["keypoints1"].numpy().astype(np.float32).reshape(-1, 1, 2)

    if len(matched_kp0) < 8:
        print("Not enough matches.")
        return None, None

    F, mask = cv2.findFundamentalMat(matched_kp0, matched_kp1, cv2.FM_RANSAC)
    if F is None or F.shape != (3, 3):
        print("Failed to compute F.")
        return None, None

    E = bot.camera_intrinsics.T @ F @ bot.camera_intrinsics
    _, R, t, inlier_mask = cv2.recoverPose(E, matched_kp0, matched_kp1, bot.camera_intrinsics)

    # Filter inliers
    inlier_mask = inlier_mask.ravel().astype(bool)
    kp0_in = matched_kp0[inlier_mask].reshape(-1, 2)
    kp1_in = matched_kp1[inlier_mask].reshape(-1, 2)

    if len(kp0_in) < 8:
        print("Not enough inliers after pose recovery.")
        return R, t  # Continue with unit scale

    # --- Triangulate ---
    P0 = bot.camera_intrinsics @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = bot.camera_intrinsics @ np.hstack((R, t))

    pts1 = kp0_in.T
    pts2 = kp1_in.T
    points_4d_hom = cv2.triangulatePoints(P0, P1, pts1, pts2)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    # Clean up 3D points
    points_3d = points_3d.T
    valid = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0) & (np.linalg.norm(points_3d, axis=1) < 10)
    points_3d = points_3d[valid]

    if len(points_3d) == 0:
        print("No valid 3D points after filtering.")
        return R, t  # fallback

    depths = points_3d[:, 2]
    scale = np.median(depths)
    t_scaled = t * scale

    print(f"Estimated scale from depth: {scale:.4f}")

    T_relative = np.eye(4)
    T_relative[:3, :3] = R
    T_relative[:3, 3] = t_scaled.flatten()

    bot.current_pose = bot.current_pose @ T_relative

    return R, t_scaled


def compute_essential_matrix(output, bot):
    # Ensure keypoints are float32 and reshaped to (N, 1, 2)
    # This (N, 1, 2) shape is often preferred by OpenCV for point sets.
    # The default output of `post_process_keypoint_matching` is (N, 2),
    # so we explicitly reshape it here.
    matched_kp0 = output["keypoints0"].numpy().astype(np.float32).reshape(-1, 1, 2)
    matched_kp1 = output["keypoints1"].numpy().astype(np.float32).reshape(-1, 1, 2)
    
    num_matches = len(matched_kp0)
    if num_matches < 8:
        print(f"WARNING: Not enough matches ({num_matches}) for Fundamental Matrix estimation. Need at least 8.")
        return None, None

    # --- Corrected cv2.findFundamentalMat call ---
    # Parameters for RANSAC:
    ransac_reproj_threshold = 3.0 # Maximum distance from point to epipolar line (in pixels)
    confidence = 0.99             # Desirable level of confidence for the estimated matrix
    max_iters = 1000              # Maximum number of RANSAC iterations (added as required by error)

    F, inliers_mask = cv2.findFundamentalMat(
        matched_kp0,
        matched_kp1,
        cv2.FM_RANSAC,
        ransac_reproj_threshold,
        confidence,
        max_iters # This argument was missing and is now explicitly added
    )
            
    if F is None:
        print("ERROR: Fundamental matrix estimation failed. Returned None.")
        return None, None
    if F.shape != (3, 3): # Sometimes F can be 9x3 for 7-point, but for 8-point/RANSAC should be 3x3
         print(f"ERROR: Fundamental matrix has unexpected shape: {F.shape}")
         return None, None

    # Convert mask to boolean array and filter
    inliers_mask = inliers_mask.ravel().astype(bool)
    matched_kp0_inliers = matched_kp0[inliers_mask].reshape(-1, 2) # Flatten back to (N, 2) for next steps
    matched_kp1_inliers = matched_kp1[inliers_mask].reshape(-1, 2)

    num_inliers = len(matched_kp0_inliers)
    if num_inliers < 5:
        print(f"WARNING: Not enough inlier matches ({num_inliers}) for Essential Matrix decomposition. Need at least 5.")
        return None, None

    # Essential Matrix from Fundamental Matrix (assuming calibrated camera)
    # Ensure K is float64 as well for matrix multiplication with float64 F
    E = bot.camera_intrinsics.T @ F @ bot.camera_intrinsics

    # Decompose Essential Matrix to get Rotation and Translation
    # cv2.recoverPose expects points to be (N, 2) or (N, 1, 2) float32
    _, R, t, _ = cv2.recoverPose(E, matched_kp0_inliers.astype(np.float32), 
                                 matched_kp1_inliers.astype(np.float32), 
                                 bot.camera_intrinsics)
    
    print("Relative Rotation (R) from previous to current frame:")
    print(R)
    print("Relative Translation (t) from previous to current frame (normalized scale):")
    print(t)
    
    T_relative = np.eye(4)
    T_relative[:3, :3] = R
    T_relative[:3, 3] = t.flatten()

    bot.current_pose = bot.current_pose @ T_relative

    return R, t

class Bot:
    def __init__(self):
        self.camera_intrinsics = np.array([[960.0000,   0.0000, 960.0000],
            [0.0000, 540.0000, 540.0000],
            [0.0000,   0.0000,   1.0000]])
        self.current_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def main():
    bot = Bot()
    image_paths = ["rendered_frames/frame_0001.png","rendered_frames/frame_0002.png"]
    images_np_batch = [cv2.imread(p) for p in image_paths]
    images_pil_batch = [Image.open(p).convert("RGB") for p in image_paths]
    outputs = run_super_glue(images_pil_batch, visualise=True)
    print(bot.current_pose)
    compute_essential_matrix(outputs, bot)
    print(bot.current_pose)

if __name__ == '__main__':
    main()