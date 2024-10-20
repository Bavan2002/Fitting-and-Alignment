# (a) Compute and Match SIFT Features Between Two Images
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the images
img1 = cv2.imread('Fitting-and-Alignment\img1.ppm', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('Fitting-and-Alignment\img5.ppm', cv2.IMREAD_GRAYSCALE)


# Check if images are loaded correctly
if img1 is None or img5 is None:
    print("Error: Could not read one of the input images. Check the file paths.")
    exit()



# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp5, des5 = sift.detectAndCompute(img5, None)

# Use FLANN based matcher to match descriptors
index_params = dict(algorithm=1, trees=5)  # FLANN parameters
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des5, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Plot the matched features
img_matches = cv2.drawMatches(img1, kp1, img5, kp5, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.title('SIFT Feature Matching')
plt.show()


# (b) Compute the Homography Using RANSAC

# Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw only inliers
matches_mask = mask.ravel().tolist()
img_inliers = cv2.drawMatches(img1, kp1, img5, kp5, good_matches, None, matchesMask=matches_mask, flags=2)

plt.imshow(img_inliers)
plt.title('Inlier Matches with RANSAC')
plt.show()

print("Homography Matrix:\n", H)


# Stitch img1.ppm onto img5.ppm

# Warp img1 to align with img5 using the computed homography
height, width = img5.shape
warped_img1 = cv2.warpPerspective(cv2.imread('Fitting-and-Alignment\img1.ppm'), H, (width, height))

# Create a canvas to hold the stitched image
stitched_image = np.maximum(warped_img1, cv2.imread('Fitting-and-Alignment\img5.ppm'))

# Display the stitched result
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
plt.title('Stitched Image')
plt.show()



