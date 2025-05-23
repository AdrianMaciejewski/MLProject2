# exercise 2.4.2

import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore
from dtuimldmtools import similarity

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = np.squeeze(mat_data["y"])
C = mat_data["C"][0, 0]
M = mat_data["M"][0, 0]
N = mat_data["N"][0, 0]

attributeNames = [name[0][0] for name in mat_data["attributeNames"]]
classNames = [cls[0] for cls in mat_data["classNames"][0]]

# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
outlier_mask = (X[:, 1] > 20) | (X[:, 7] > 10) | (X[:, 10] > 200)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask, :]
y = y[valid_mask]
N = len(y)
Xnorm = zscore(X, ddof=1)

## Next we plot a number of atttributes
Attributes = [1, 4, 5, 6]
NumAtr = len(Attributes)

plt.figure(figsize=(12, 12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1 * NumAtr + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plt.plot(X[class_mask, Attributes[m2]], X[class_mask, Attributes[m1]], ".")
            if m1 == NumAtr - 1:
                plt.xlabel(attributeNames[Attributes[m2]])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[Attributes[m1]])
            else:
                plt.yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
plt.legend(classNames)
plt.show()

print("Ran Exercise 2.4.2")