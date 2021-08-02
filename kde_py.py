from re import L
import numpy
import pandas as pd
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
import numpy as np
from scipy import integrate, stats, interpolate

from numpy import trapz
from matplotlib import pyplot as plt


def integ(data):
    out_data = [data[0]]
    k = 1
    d_x = (out_data[-1] - out_data[0])/len(out_data)
    tr = trapz(out_data, dx = d_x)
    print('fistr area: ' ,tr, '\n', 'out_data: ', out_data)

    while len(out_data) != len(data):
        out_data.append(data[k])
        
        inner_d_x = (out_data[-1] - out_data[0])/len(out_data)
        
        inner_tr = trapz(out_data, dx = inner_d_x)
        print(out_data, '\n\n', 'Area: ',inner_tr, '\n\n----\n')
        k+=1

    return out_data



"""
    #Тестовые(промежуточные) данные
    mid_data = [-0.7, -0.2]
    mid_dx = (mid_data[-1] - mid_data[0])/len(mid_data)
    mid_area = trapz(mid_data, dx=mid_dx)
    # print('mid_dx: ' , mid_dx ,'\n' , 'Mid_area: ' ,mid_area)



    #Данные
    data = [-0.7, -0.2, -0.2, 0.0, 0.1, 0.8, 1.1, 1.2, 1.4]
    d_x = (data[-1] - data[0])/len(data)
    area = trapz(data, dx=d_x)
"""


df = pd.read_excel ('book.xlsx')
df_list = df.to_numpy()
tr_dflist = df_list.transpose()

tr_dflist = tr_dflist.tolist()


col_list = tr_dflist[1]

out_y = []


kde_col_list = stats.gaussian_kde(col_list)

for i in range(len(col_list)):
    area_kde_col_list = kde_col_list.integrate_box_1d(col_list[0], col_list[i])
    out_y.append(area_kde_col_list)
    # print(area_kde_col_list)

f = interpolate.interp1d(col_list, out_y)

xnew = np.arange(col_list[0], col_list[-1])
ynew = f(xnew)


plt.plot(col_list, out_y, 'o', xnew, ynew, '-')
plt.show()


# kde1 = stats.gaussian_kde(tr_dflist[1])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ll, np.zeros(ll.shape), 'ro', ms=5)

# #filling X-axis
# # x_eval = np.linspace(-10, 10, num=200)
# x = np.linspace(ll.min(), ll.max(), 120)

# ax.plot(x, kde1(x), 'k-', label="Scott's Rule")


# j = 0


# for i in tr_dflist:
#     integ(i)
#     # ll.append(tr_dflist[i])
#     # kde1 = stats.gaussian_kde(ll)

#     # # area = kde1.integrate_box_1d(i[0], i[-1])

#     j+=1

# print(j)














# dx = (data[-1] - data[0])/len(data)
# trapez = d_x/2*(data[0]+2*data[1]+2*data[2]+2*data[3]+2*data[4]+2*data[5]+2*data[6]+2*data[7]+data[8])


# x, y = FFTKDE(bw="silverman").fit(data).evaluate()

# # Use scipy to interplate and evaluate on arbitrary grid
# x_grid = np.array([-2.5, -2, -1, 0, 0.5, 1, 1.5, 1.75, 2, 2.25, 2.5])
# f = interp1d(x, y, kind="linear", assume_sorted=True)
# y_grid = f(x_grid)

# # Plot the resulting KDEs
# plt.scatter(data, np.zeros_like(data), marker='o', label="Data")
# plt.plot(x, y)
# plt.plot(x_grid, y_grid, '-o', label="")
# plt.tight_layout(); plt.legend(loc='upper left')

# plt.show()