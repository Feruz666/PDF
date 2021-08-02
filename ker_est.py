import pandas as pd
from scipy import integrate, stats, interpolate
import numpy as np

from numpy import trapz
from matplotlib import pyplot as plt

# df = pd.read_excel ('book.xlsx')
# df_list = df.to_numpy()
# tr_dflist = df_list.transpose()


x1 = np.array([1, 2,2.1,2.3,2.4,2.6, 3, 4, 5,6], dtype=np.float64)  
kde1 = stats.gaussian_kde(x1)

area = kde1.integrate_box_1d(x1[0], x1[-1])


print(area)




fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, np.zeros(x1.shape), 'ro', ms=5)

#filling X-axis
# x_eval = np.linspace(-10, 10, num=200)
x = np.linspace(x1.min(), x1.max(), 20)

ax.plot(x, kde1(x), 'k-', label="Scott's Rule")

# и на том примере, который вы мне показывали на доске, там плотность была высока посередине, там можно было посчитать интегралы 50%, 75%.
# Как

plt.show()