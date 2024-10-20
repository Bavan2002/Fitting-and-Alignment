import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
im = cv2.imread('images/the_berry_farms_sunflower_field.jpeg', cv2.IMREAD_COLOR)
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with varying sigma
sigma_values = range(1, 21, 2)
blobs = []

for sigma in sigma_values:
    LoG = cv2.GaussianBlur(gray_im, (0, 0), sigma)
    LoG = cv2.Laplacian(LoG, cv2.CV_64F)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(LoG)
    blobs.append((maxLoc, sigma))

# Plot detected blobs as circles
for center, sigma in blobs:
    cv2.circle(im, center, int(sigma * np.sqrt(2)), (0, 255, 0), 2)

# Show result
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title('Blob Detection using LoG')
plt.show()
