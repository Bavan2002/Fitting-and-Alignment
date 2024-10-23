from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Constants
NUM_SAMPLES = 100
half_samples = NUM_SAMPLES // 2

# Circle parameters
circle_radius = 10
circle_center_x, circle_center_y = 2, 3
noise_factor = circle_radius / 20  # Slight adjustment in noise scaling
angles = np.random.uniform(0, 2 * np.pi, half_samples)
circle_noise = noise_factor * np.random.randn(half_samples)

# Generate circle points
circle_x = circle_center_x + (circle_radius + circle_noise) * np.cos(angles)
circle_y = circle_center_y + (circle_radius + circle_noise) * np.sin(angles)
circle_points = np.hstack((circle_x.reshape(half_samples, 1), circle_y.reshape(half_samples, 1)))

# Line parameters
line_slope, line_intercept, line_variation = -1, 2, 1
line_x = np.linspace(-12, 12, half_samples)
line_y = line_slope * line_x + line_intercept + line_variation * np.random.randn(half_samples)
line_points = np.hstack((line_x.reshape(half_samples, 1), line_y.reshape(half_samples, 1)))

# Combine all points into a single dataset
all_points = np.vstack((circle_points, line_points))

# Custom plotting function
def plot_dataset(separate_view=True):
    """Plot the generated point set with optional separation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if separate_view:
        ax.scatter(line_points[:, 0], line_points[:, 1], color='blue', label='Line Points')
        ax.scatter(circle_points[:, 0], circle_points[:, 1], color='green', label='Circle Points')
    else:
        ax.scatter(all_points[:, 0], all_points[:, 1], color='purple', label='Combined Points')

    # Draw the ground truth circle
    ground_circle = plt.Circle((circle_center_x, circle_center_y), circle_radius, 
                               color='orange', fill=False, linewidth=2, label='Ground Truth Circle')
    ax.add_patch(ground_circle)
    ax.plot(circle_center_x, circle_center_y, 'x', color='orange')

    # Plot the ground truth line
    x_min, x_max = ax.get_xlim()
    line_range = np.array([x_min, x_max])
    line_y_values = line_slope * line_range + line_intercept
    ax.plot(line_range, line_y_values, color='red', linestyle='--', linewidth=2, label='Ground Truth Line')

    ax.legend()
    plt.grid(True)
    return ax

# Plot the dataset
plot_dataset()
plt.show()



def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    (x1, y1), (x2, y2) = point1, point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def compute_line_parameters(pt1, pt2):
    """
    Compute the line equation parameters: a*x + b*y = d,
    given two points on the line.
    """
    (x1, y1), (x2, y2) = pt1, pt2
    a_coeff = y2 - y1
    b_coeff = -(x2 - x1)
    d_val = a_coeff * x1 + b_coeff * y1

    # Assert line equation holds for the second point
    assert abs(a_coeff * x2 + b_coeff * y2 - d_val) < 1e-8

    # Normalize parameters to ensure a unit normal vector
    norm_factor = (a_coeff ** 2 + b_coeff ** 2) ** 0.5
    a_coeff, b_coeff, d_val = a_coeff / norm_factor, b_coeff / norm_factor, d_val / norm_factor
    return a_coeff, b_coeff, d_val

def compute_circle_parameters(pt1, pt2, pt3):
    """
    Calculate the center (cx, cy) and radius (r) of a circle
    that passes through three given points.
    """
    (x1, y1), (x2, y2), (x3, y3) = pt1, pt2, pt3

    # Midpoints of two line segments
    midpoint1_x, midpoint1_y = (x1 + x2) / 2, (y1 + y2) / 2
    midpoint2_x, midpoint2_y = (x1 + x3) / 2, (y1 + y3) / 2

    # Slopes of the perpendicular bisectors
    slope1 = -(x2 - x1) / (y2 - y1)
    slope2 = -(x3 - x1) / (y3 - y1)

    # Intercepts of the perpendicular bisectors
    intercept1 = midpoint1_y - slope1 * midpoint1_x
    intercept2 = midpoint2_y - slope2 * midpoint2_x

    # Solving for the intersection point of the two bisectors (circle center)
    center_x = (intercept2 - intercept1) / (slope1 - slope2)
    center_y = slope1 * center_x + intercept1

    # Verify the intersection point lies on both bisectors
    assert abs(slope2 * center_x + intercept2 - center_y) < 1e-8

    # Calculate the radius of the circle
    radius = calculate_distance((center_x, center_y), pt1)

    # Verify the radius is consistent for all three points
    assert abs(calculate_distance((center_x, center_y), pt2) - radius) < 1e-8

    return center_x, center_y, radius


# Calculate squared error for line fitting
def line_fit_error(parameters, *data):
    """
    Compute the total squared error for a line given the parameters.
    Parameters are optimized to minimize this error.
    """
    a_coeff, b_coeff, d_value = parameters
    point_indices, points = data
    error_sum = np.sum((a_coeff * points[point_indices, 0] + 
                        b_coeff * points[point_indices, 1] - d_value) ** 2)
    return error_sum

# Calculate squared error for circle fitting
def circle_fit_error(parameters, *data):
    """
    Compute the squared error for circle fitting.
    The goal is to minimize the difference between the expected and actual radius.
    """
    center_x, center_y, radius = parameters
    point_indices, points = data
    error_sum = np.sum((calculate_distance((center_x, center_y), 
                                           (points[point_indices, 0], points[point_indices, 1])) - radius) ** 2)
    return error_sum

# Check which points are inliers for a line model
def find_line_inliers(parameters, threshold, points):
    """
    Identify the inliers for a line model given a threshold.
    """
    a_coeff, b_coeff, d_value = parameters
    residuals = np.abs(a_coeff * points[:, 0] + b_coeff * points[:, 1] - d_value)
    return np.where(residuals < threshold)

# Check which points are inliers for a circle model
def find_circle_inliers(parameters, threshold, points):
    """
    Identify the inliers for a circle model based on radial error.
    """
    center_x, center_y, radius = parameters
    residuals = np.abs(calculate_distance((center_x, center_y), 
                                          (points[:, 0], points[:, 1])) - radius)
    return np.where(residuals < threshold)

# Constraint to normalize the line parameters
def line_constraint(parameters):
    """
    Ensure the line parameters represent a unit normal vector.
    """
    a_coeff, b_coeff, d_value = parameters
    return (a_coeff ** 2 + b_coeff ** 2) ** 0.5 - 1

# Define the constraint dictionary for the optimizer
constraint_def = {'type': 'eq', 'fun': line_constraint}

# Perform least squares fitting for a line
def fit_line_least_squares(inlier_indices, initial_guess, points):
    """
    Fit a line using least squares optimization with constraints.
    """
    result = minimize(fun=line_fit_error, 
                      x0=initial_guess, 
                      args=(inlier_indices, points), 
                      constraints=constraint_def, 
                      tol=1e-6)
    print(f"Fitted Line Parameters: {result.x}, Error: {result.fun}")
    return result

# Perform least squares fitting for a circle
def fit_circle_least_squares(inlier_indices, initial_guess, points):
    """
    Fit a circle using least squares optimization without constraints.
    """
    result = minimize(fun=circle_fit_error, 
                      x0=initial_guess, 
                      args=(inlier_indices, points), 
                      tol=1e-6)
    print(f"Fitted Circle Parameters: {result.x}, Error: {result.fun}")
    return result


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Generate Circle and Line Points ---

# Circle parameters
circle_radius = 10
circle_center_x, circle_center_y = 2, 3
noise_factor = circle_radius / 20  # Adjust noise for the circle points
num_circle_points = 50

angles = np.random.uniform(0, 2 * np.pi, num_circle_points)
circle_noise = noise_factor * np.random.randn(num_circle_points)

circle_x = circle_center_x + (circle_radius + circle_noise) * np.cos(angles)
circle_y = circle_center_y + (circle_radius + circle_noise) * np.sin(angles)
circle_points = np.hstack((circle_x.reshape(-1, 1), circle_y.reshape(-1, 1)))

# Line parameters
line_slope, line_intercept, line_variation = -1, 2, 1
num_line_points = 50

line_x = np.linspace(-12, 12, num_line_points)
line_y = line_slope * line_x + line_intercept + line_variation * np.random.randn(num_line_points)
line_points = np.hstack((line_x.reshape(-1, 1), line_y.reshape(-1, 1)))

# Combine all points into a single dataset
X = np.vstack((circle_points, line_points))  # Dataset used in RANSAC

# --- Plot the Initial Data ---
def plot_initial_data():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(circle_points[:, 0], circle_points[:, 1], color='green', label='Circle Points')
    ax.scatter(line_points[:, 0], line_points[:, 1], color='blue', label='Line Points')

    circle = plt.Circle((circle_center_x, circle_center_y), circle_radius, 
                        color='orange', fill=False, linewidth=2, label='True Circle')
    ax.add_patch(circle)

    x_min, x_max = ax.get_xlim()
    y_range = line_slope * np.array([x_min, x_max]) + line_intercept
    ax.plot([x_min, x_max], y_range, 'r--', linewidth=2, label='True Line')

    plt.legend()
    plt.grid(True)

    # Ensure the Axes object is returned
    return ax


plot_initial_data()

# --- Utility Functions ---

def calculate_distance(pt1, pt2):
    """Calculate the Euclidean distance between two points."""
    (x1, y1), (x2, y2) = pt1, pt2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def compute_line_parameters(pt1, pt2):
    """Compute the line parameters: a*x + b*y = d."""
    (x1, y1), (x2, y2) = pt1, pt2
    a_coeff, b_coeff = y2 - y1, -(x2 - x1)
    d_value = a_coeff * x1 + b_coeff * y1

    norm = np.sqrt(a_coeff ** 2 + b_coeff ** 2)
    return a_coeff / norm, b_coeff / norm, d_value / norm

def compute_circle_parameters(pt1, pt2, pt3):
    """Compute the center and radius of a circle given three points."""
    (x1, y1), (x2, y2), (x3, y3) = pt1, pt2, pt3

    mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
    mid2_x, mid2_y = (x1 + x3) / 2, (y1 + y3) / 2

    slope1, slope2 = -(x2 - x1) / (y2 - y1), -(x3 - x1) / (y3 - y1)
    intercept1, intercept2 = mid1_y - slope1 * mid1_x, mid2_y - slope2 * mid2_x

    center_x = (intercept2 - intercept1) / (slope1 - slope2)
    center_y = slope1 * center_x + intercept1
    radius = calculate_distance((center_x, center_y), pt1)

    return center_x, center_y, radius

def find_line_inliers(params, threshold, points):
    """Identify line inliers based on the given threshold."""
    a, b, d = params
    residuals = np.abs(a * points[:, 0] + b * points[:, 1] - d)
    return np.where(residuals < threshold)

def find_circle_inliers(params, threshold, points):
    """Identify circle inliers based on radial error."""
    cx, cy, r = params
    residuals = np.abs(calculate_distance((cx, cy), (points[:, 0], points[:, 1])) - r)
    return np.where(residuals < threshold)

# --- RANSAC for Line Fitting ---
iterations = 100
min_line_points = 2
line_threshold = 1.0
min_inliers_line = 0.4 * X.shape[0]

best_line_params = None
best_line_inliers = None
smallest_line_error = float('inf')

# Inside RANSAC loop for line fitting
for _ in range(iterations):
    sample_indices = np.random.choice(X.shape[0], size=min_line_points, replace=False)
    line_params = compute_line_parameters(X[sample_indices[0]], X[sample_indices[1]])
    inliers = find_line_inliers(line_params, line_threshold, X)[0]

    if len(inliers) >= min_inliers_line:
        result = minimize(lambda p: np.sum((p[0] * X[inliers, 0] + 
                                            p[1] * X[inliers, 1] - p[2]) ** 2),
                          x0=line_params, constraints={'type': 'eq', 'fun': lambda p: np.sqrt(p[0]**2 + p[1]**2) - 1})

        if result.fun < smallest_line_error:
            smallest_line_error = result.fun
            best_line_params = result.x
            best_line_inliers = inliers
            # Store the best sample indices
            best_line_samples = sample_indices


print(f'Best Line: {best_line_params}, Error: {smallest_line_error}')

# --- RANSAC for Circle Fitting ---
remaining_indices = np.setdiff1d(np.arange(X.shape[0]), best_line_inliers)
X_remaining = X[remaining_indices]

min_circle_points = 3
circle_threshold = 1.2
min_inliers_circle = 0.4 * X.shape[0]

best_circle_params = None
best_circle_inliers = None
smallest_circle_error = float('inf')

# RANSAC loop for circle fitting
for _ in range(iterations):
    sample_indices = np.random.choice(X_remaining.shape[0], size=min_circle_points, replace=False)
    circle_params = compute_circle_parameters(*X_remaining[sample_indices])
    inliers = find_circle_inliers(circle_params, circle_threshold, X_remaining)[0]

    if len(inliers) >= min_inliers_circle:
        result = minimize(lambda p: np.sum((calculate_distance((p[0], p[1]), 
                                                               (X_remaining[inliers, 0], X_remaining[inliers, 1])) - p[2]) ** 2),
                          x0=circle_params)

        if result.fun < smallest_circle_error:
            smallest_circle_error = result.fun
            best_circle_params = result.x
            best_circle_inliers = inliers
            # Store the best sample indices
            best_circle_samples = sample_indices


print(f'Best Circle: {best_circle_params}, Error: {smallest_circle_error}')

# Ensure ax is assigned correctly
ax = plot_initial_data()

# Get the range for plotting the line
x_min, x_max = ax.get_xlim()
x_range = np.array([x_min, x_max])
A, B, D = best_line_params  # Unpack line parameters

# Calculate the y-values for the line
y_range = (D - A * x_range) / B
ax.plot(x_range, y_range, color='red', label='RANSAC Fitted Line')

# Plot the line inliers
ax.scatter(X[best_line_inliers, 0], X[best_line_inliers, 1], color='pink', label='Line Inliers')

# Plot the best sample points used for the line
ax.scatter(X[best_line_samples, 0], X[best_line_samples, 1], color='black', label='Best Sample for Line')

# Plot the fitted circle
x_center, y_center, radius = best_circle_params
ransac_circle = plt.Circle((x_center, y_center), radius, color='black', fill=False, label='RANSAC Fitted Circle')
ax.add_patch(ransac_circle)
ax.plot(x_center, y_center, '+', color='black')

# Plot circle inliers and best sample points for the circle
ax.scatter(X_remaining[best_circle_inliers, 0], X_remaining[best_circle_inliers, 1], color='yellow', label='Circle Inliers')
ax.scatter(X_remaining[best_circle_samples, 0], X_remaining[best_circle_samples, 1], color='red', label='Best Sample for Circle')

# Show the legend and the final plot
plt.legend()
plt.show()
