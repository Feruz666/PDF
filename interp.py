from scipy.interpolate import interp1d
import numpy as np


x = np.linspace(0, 10, num=11, endpoint=True)

y = np.linspace(1,20, len(x))

f = interp1d(x, y)

f2 = interp1d(x, y, kind='cubic')
print(f2)

xnew = np.linspace(0, 10, num=41, endpoint=True)

import matplotlib.pyplot as plt

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')

plt.legend(['data', 'linear', 'cubic'], loc='best')

plt.show()