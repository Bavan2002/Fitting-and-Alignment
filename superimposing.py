import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# Read and prepare images
img3 = cv.imread(r"C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\2\Fitting-and-Alignment\img3.png")
img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

logo = cv.imread(r"C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\2\Fitting-and-Alignment\img4.jpg")
logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)

# Destination points for transformation
dst_points = np.array([(127, 577), (428,524), (471,65),(132, 51)])

def superimpose(image, logo, dst_points, beta=0.3, alpha=1):
    y, x, _ = logo.shape
    src_points = np.array([(0, y), (x, y), (x, 0), (0, 0)])  # bl, br, tr, tl
    # Pad logo to match the size of the base image if needed
    if logo.shape[0] < image.shape[0]:
        logo = np.pad(logo, ((0, image.shape[0] - logo.shape[0]), (0, 0), (0, 0)), 'constant')

    if logo.shape[1] < image.shape[1]:
        logo = np.pad(logo, ((0, 0), (0, image.shape[1] - logo.shape[1]), (0, 0)), 'constant')
    # Estimate the projective transformation
    tform = transform.ProjectiveTransform()
    tform.estimate(src_points, dst_points)
    # Apply the inverse transformation to the logo
    tf_img = transform.warp(logo, tform.inverse, output_shape=image.shape)
    tf_img = (tf_img * 255).astype(np.uint8)  # Convert to uint8
    # Blend the transformed logo with the base image
    dst = cv.addWeighted(image, alpha, tf_img, beta, 0) 
    return dst

# Apply the superimpose function
dst = superimpose(img3, logo, dst_points, beta=0.3, alpha=0.9)

# Plot the results
plt.figure(figsize=(15, 5))

# Original image with points
plt.subplot(1, 3, 1)
plt.imshow(img3)
plt.scatter(dst_points[:, 0], dst_points[:, 1], c='red', marker='x')
plt.title("Image 1 with Selected Points")
plt.axis('off')

# Logo image
plt.subplot(1, 3, 2)
plt.imshow(logo)
plt.title("Image 2 (Logo)")
plt.axis('off')

# Final superimposed image
plt.subplot(1, 3, 3)
plt.imshow(dst)
plt.title("Final Image with Superimposed Logo")
plt.axis('off')

plt.tight_layout()
plt.show()
