import numpy as np
import matplotlib.pyplot as plt

# Generate noisy data
np.random.seed(0)
N = 100
X = np.linspace(-12, 12, N)
Y = 2 * X + 1 + np.random.randn(N)

# RANSAC for line fitting
max_inliers = 0
best_line = None

for _ in range(1000):
    sample_indices = np.random.choice(N, 2, replace=False)
    x_sample, y_sample = X[sample_indices], Y[sample_indices]
    coeffs = np.polyfit(x_sample, y_sample, 1)
    
    distances = np.abs(Y - (coeffs[0] * X + coeffs[1]))
    inliers = np.sum(distances < 1.0)

    if inliers > max_inliers:
        max_inliers = inliers
        best_line = coeffs

# Plot result
plt.scatter(X, Y, label='Data Points')
plt.plot(X, best_line[0] * X + best_line[1], 'r', label='RANSAC Line Fit')
plt.legend()
plt.show()
