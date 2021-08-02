from sklearn.neighbors import KernelDensity
import numpy as np
X = np.array([-7, -5, 1, 1.2, 1.5, 1.6, 1.8, 2, 2.5, 4, 5])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
print(kde.score_samples(X))