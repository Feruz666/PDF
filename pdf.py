# # from scipy import stats
# # import numpy as np
# # import matplotlib.pyplot as plt

# # x = np.linspace(-10, 10, 20)

# # x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float64)
# # kde1 = stats.gaussian_kde(x1)
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
# # x_eval = np.linspace(-10, 10, num=200)
# # ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")


# # randnums= np.random.rand(5,3)
# # print(randnums, '\n')
# # for i in randnums:
# #     i.sort()
# # print(randnums)

from typing import final
from matplotlib import pyplot as plt
# import numpy as np
# import seaborn as sns

# x = np.random.normal(0, 1, 100)
# mean = x.mean()
# std = x.std()
# q1, median, q3 = np.percentile(x, [25, 50, 75])
# iqr = q3 - q1

# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

# medianprops = dict(linestyle='-', linewidth=2, color='yellow')
# sns.boxplot(x=x, color='lightcoral', saturation=1, medianprops=medianprops,
#             flierprops={'markerfacecolor': 'mediumseagreen'}, whis=1.5, ax=ax1)

# ticks = [mean + std * i for i in range(-4, 5)]
# ticklabels = [f'${i}\\sigma$' for i in range(-4, 5)]
# ax1.set_xticks(ticks)
# ax1.set_xticklabels(ticklabels)
# ax1.set_yticks([])
# ax1.tick_params(labelbottom=True)
# ax1.set_ylim(-1, 1.5)
# ax1.errorbar([q1, q3], [1, 1], yerr=[-0.2, 0.2], color='black', lw=1)
# ax1.text(q1, 0.6, 'Q1', ha='center', va='center', color='black')
# ax1.text(q3, 0.6, 'Q3', ha='center', va='center', color='black')
# ax1.text(median, -0.6, 'median', ha='center', va='center', color='black')
# ax1.text(median, 1.2, 'IQR', ha='center', va='center', color='black')
# ax1.text(q1 - 1.5*iqr, 0.4, 'Q1 - 1.5*IQR', ha='center', va='center', color='black')
# ax1.text(q3 + 1.5*iqr, 0.4, 'Q3 + 1.5*IQR', ha='center', va='center', color='black')
# # ax1.vlines([q1 - 1.5*iqr, q1, q3, q3 + 1.5*iqr], 0, -2, color='darkgrey', ls=':', clip_on=False, zorder=0)

# sns.kdeplot(x, ax=ax2)
# kdeline = ax2.lines[0]
# xs = kdeline.get_xdata()
# ys = kdeline.get_ydata()

# ylims = ax2.get_ylim()
# ax2.fill_between(xs, 0, ys, color='mediumseagreen')
# ax2.fill_between(xs, 0, ys, where=(xs >= q1 - 1.5*iqr) & (xs <= q3 + 1.5*iqr), color='skyblue')
# ax2.fill_between(xs, 0, ys, where=(xs >= q1) & (xs <= q3), color='lightcoral')
# # ax2.vlines([q1 - 1.5*iqr, q1, q3, q3 + 1.5*iqr], 0, 100, color='darkgrey', ls=':', zorder=0)
# ax2.set_ylim(0, ylims[1])
# plt.show()

import pandas as pd
from scipy import integrate, stats
import numpy as np

from numpy import trapz


# x = np.linspace(-9,10, num=20)
# y = x
# y_int = integrate.cumtrapz(y,x, initial=0)
# plt.plot(x, y_int, 'ro',)
np.set_printoptions(suppress=True)


df = pd.read_excel ('book.xlsx')
df_list = df.to_numpy()
tr_dflist = df_list.transpose()

final_list = []
k = 0
for i in range (len(tr_dflist)):
    tr_dflist[i].sort()
    # new_list = tr_dflist[i][int(len(tr_dflist[i]) * 0.25) : int(len(tr_dflist[i]) * 0.75)]
    # area = integrate.simps(new_list, dx=1e-6)
    # final_list.append(area)
    kde1 = stats.gaussian_kde(tr_dflist[i])
    area1 = kde1.integrate_box_1d(tr_dflist[i][0], tr_dflist[i][-1])




first_l = tr_dflist[1]
#Converting to float sorted list
# for i in range(4):
    # print(float(tr_dflist[1][i]))


new_list = first_l[int(len(first_l) * 0.25) : int(len(first_l) * 0.75)]


listlist = np.linspace(1, 100, num = 100)
new_test_list = listlist[int(len(listlist) * 0.25) : int(len(listlist) * 0.75)]


"""

0.710555600700074
0.718687326937075
1.72762556396831
2.51838375780481

np.around([0.37, 1.64], decimals=1)
[7.10555601e-01 7.18687327e-01 1.72762556e+00 2.51838376e+00
 3.44754731e+00 3.77569961e+00 4.84296979e+00 4.85663940e+00
 5.19769321e+00 5.72408857e+00 7.37217930e+00 1.01595967e+01
 1.19585856e+01 1.22139934e+01 1.25820277e+01 1.26538142e+01
 1.48635655e+01 1.52390249e+01 1.54682191e+01 1.80370052e+01
 1.90111988e+01 2.06255002e+01 2.46423961e+01 2.62363882e+01
 2.67231968e+01 2.84251516e+01 3.09075535e+01 3.13148279e+01
 3.21019704e+01 3.34870462e+01 3.46620371e+01 3.53716870e+01
 3.63097592e+01 3.76143960e+01 3.90528477e+01 4.16197939e+01
 4.21418602e+01 4.90586912e+01 5.01382538e+01 5.72765477e+01
 5.80153756e+01 6.37600754e+01 6.53475832e+01 7.68708042e+01
 8.50778782e+01 9.71089979e+01 1.00832408e+02 1.09714243e+02
 1.10469152e+02 1.13529690e+02 1.14148829e+02 1.15275161e+02
 1.43045681e+02 1.44029368e+02 1.44146120e+02 1.47660045e+02
 1.47679173e+02 1.50061464e+02 1.57918265e+02 1.59218454e+02
 1.80286235e+02 1.87618332e+02 1.90006579e+02 1.99692375e+02
 2.01542808e+02 2.21029154e+02 2.40447575e+02 2.58133397e+02
 2.79265250e+02 2.79980107e+02 3.05449610e+02 3.09928675e+02
 3.31718338e+02 3.39304622e+02 4.03261550e+02 4.50430802e+02
 4.52519900e+02 4.99262919e+02 6.18640554e+02 6.28577735e+02
 6.62542171e+02 6.87730609e+02 7.14563101e+02 7.70865299e+02
 7.74403347e+02 7.87361294e+02 8.12577006e+02 8.23362340e+02
 8.73007748e+02 8.84459437e+02 9.69393477e+02 1.01213305e+03
 1.01788951e+03 1.07586480e+03 1.16105580e+03 1.22347075e+03
 1.22455495e+03 1.26025346e+03 1.37424945e+03 1.37718330e+03
 1.44996890e+03 1.47882462e+03 1.57200985e+03 1.62625533e+03
 1.64758419e+03 1.72879748e+03 1.93946718e+03 2.01498418e+03
 2.11911148e+03 2.15422584e+03 2.15462488e+03 2.18567123e+03
 2.22823266e+03 2.33167631e+03 2.35661988e+03 2.56532404e+03
 2.59093941e+03 2.77722353e+03 3.15932502e+03 3.23592403e+03]

 """