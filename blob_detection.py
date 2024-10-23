import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def compute_laplacian_gaussian(sigma_val):
    """Generate the Laplacian of Gaussian (LoG) kernel for a given sigma."""
    half_width = round(3 * sigma_val)  # Calculate half the width of the kernel
    X, Y = np.meshgrid(np.arange(-half_width, half_width + 1),
                       np.arange(-half_width, half_width + 1))

    # Compute the LoG filter
    log_kernel = ((X**2 + Y**2) / (2 * sigma_val**2) - 1) * \
                 np.exp(-(X**2 + Y**2) / (2 * sigma_val**2)) / (np.pi * sigma_val**4)
    return log_kernel

def identify_local_maxima(log_image, sigma_val):
    """Detect local maxima from the filtered image."""
    detected_coordinates = []
    (height, width) = log_image.shape
    offset = 1  # Size of the neighborhood to check

    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            neighborhood = log_image[i-offset:i+offset+1, j-offset:j+offset+1]
            max_value = np.max(neighborhood)
            if max_value >= 0.1:  # Adjusted threshold for detection
                x, y = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
                detected_coordinates.append((i + x - offset, j + y - offset))  # Coordinate correction

    return set(detected_coordinates)

# Load and preprocess the image
sunflower_image = cv.imread(r'C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\2\Fitting-and-Alignment\the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
sunflower_gray = cv.cvtColor(sunflower_image, cv.COLOR_BGR2GRAY) / 255.0

fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # Adjusted figure size

# Loop through different sigma values
for idx, ax in enumerate(axes.flatten(), start=1):
    sigma_val = idx / 1.414  # Varying sigma values
    log_kernel = sigma_val**2 * compute_laplacian_gaussian(sigma_val)
    log_filtered_image = np.square(cv.filter2D(sunflower_gray, -1, log_kernel))

    detected_points = identify_local_maxima(log_filtered_image, sigma_val)

    # Display the results for each sigma
    ax.imshow(log_filtered_image, cmap='gray')
    ax.set_title(f'Sigma = {round(sigma_val, 2)}')

    for (x, y) in detected_points:
        circle = plt.Circle((y, x), sigma_val * 1.414, color='blue', linewidth=1, fill=False)
        ax.add_patch(circle)

plt.tight_layout()
plt.axis('off')
plt.show()

# Display original and blob-detected images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display original image
axes[0].imshow(cv.cvtColor(sunflower_image, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Sunflower Field")
axes[0].axis('off')

# Display the grayscale image with detected blobs
ax = axes[1]
ax.imshow(sunflower_gray, cmap='gray')
ax.axis('off')

# Use a color palette to assign different colors to circles
color_options = list(mcolors.BASE_COLORS)  # Adjusted color palette

circles = []
circle_labels = []

for idx in range(1, 11):
    sigma_val = idx / 1.414
    log_kernel = sigma_val**2 * compute_laplacian_gaussian(sigma_val)
    log_filtered_image = np.square(cv.filter2D(sunflower_gray, -1, log_kernel))

    detected_points = identify_local_maxima(log_filtered_image, sigma_val)

    for (x, y) in detected_points:
        color_choice = color_options[idx % len(color_options)]  # Cycling through colors
        circle = plt.Circle((y, x), sigma_val * 1.414, color=color_choice, linewidth=1, fill=False)
        ax.add_patch(circle)

    circles.append(circle)
    circle_labels.append(f'Sigma = {round(sigma_val, 2)}')

ax.legend(circles, circle_labels, loc='best', fontsize=8)
ax.set_title("Blob Detection with Varying Sigma Values")

plt.tight_layout()
plt.show()
