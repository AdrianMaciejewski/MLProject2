# exercise 11.2.2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# Draw samples from mixture of gaussians (as in exercise 11.1.1)
N = 1000
M = 1
x = np.linspace(-10, 10, 50)
X = np.empty((N, M))
m = np.array([1, 3, 6])
s = np.array([1, 0.5, 2])
c_sizes = np.random.multinomial(N, [1.0 / 3, 1.0 / 3, 1.0 / 3])
for c_id, c_size in enumerate(c_sizes):
    X[
        c_sizes.cumsum()[c_id] - c_sizes[c_id] : c_sizes.cumsum()[c_id], :
    ] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size, M))


# x-values to evaluate the KDE
xe = np.linspace(-10, 10, 100)

# Compute kernel density estimate
kde = gaussian_kde(X.ravel())

# Plot kernel density estimate
plt.figure(figsize=(6, 7))
plt.subplot(2, 1, 1)
plt.hist(X, x)
plt.title("Data histogram")
plt.subplot(2, 1, 2)
plt.plot(xe, kde.evaluate(xe))
plt.title("Kernel density estimate")
plt.show()

print("Ran Exercise 11.2.2")
