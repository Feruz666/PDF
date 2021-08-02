import numpy as np, scipy.stats as st
from numpy.core.fromnumeric import size
from numpy.core.defchararray import count
from numpy.lib.arraypad import pad
from scipy import stats
from scipy.stats import sem, t
import pandas as pd
from matplotlib import pyplot as plt
import scipy.integrate as integrate 
from sklearn.neighbors import KernelDensity
from scipy import interpolate
np.set_printoptions(suppress=True)


np.random.seed(1)

df = pd.read_excel ('book.xlsx', sheet_name='Лист1', usecols="E")
df_full = pd.read_excel('book.xlsx')
df2 = pd.read_excel ('book.xlsx')
df_list = df2.to_numpy()
tr_dflist = df_list.transpose()

tr_dflist = tr_dflist.tolist()

col_list = tr_dflist[3]
col_list = np.array(col_list)
kde_col_list = stats.gaussian_kde(col_list)

col_list.sort()



y = []
x = []

m_list = []
start_50_l = []
end_50_l = []
start_75_l = []
end_75_l = []
start_95_l = []
end_95_l = []
i=1
for i in range(10):
# for i in range(70,len(tr_dflist), 1):
    
    m = np.mean(tr_dflist[i])
    start_50, end_50 = np.percentile(tr_dflist[i], 25), np.percentile(tr_dflist[i], 75)
    start_75, end_75 = np.percentile(tr_dflist[i], 12.5), np.percentile(tr_dflist[i], 87.5)
    start_95, end_95 = np.percentile(tr_dflist[i], 2.5), np.percentile(tr_dflist[i], 97.5)

    if m <= 0.:
        m = 0
    if start_50 <= 0.:
        start_50 = 0
    if end_50 <= 0.:
        end_50 = 0
    if start_75 <= 0.:
        start_75 = 0
    if end_75 <= 0.:
        end_75 = 0
    if start_95 <= 0.:
        start_95 = 0
    if  end_95 <= 0.:
        end_95 = 0

    
    m_list.append(m)
    start_50_l.append(start_50)
    end_50_l.append(end_50)
    start_75_l.append(start_75)
    end_75_l.append(end_75)
    start_95_l.append(start_95)
    end_95_l.append(end_95)


#Mean
x = np.arange(len(m_list))
M_BSpline = interpolate.make_interp_spline(x, m_list)
xm_new = np.arange(x[0], x[-1], 0.1)
ym_new = M_BSpline(xm_new)

for i in range(len(ym_new)):
    if ym_new[i] <= 0:
        ym_new[i] = 0

#Start_50
S50_BSpline = interpolate.make_interp_spline(x, start_50_l)
xs50_new = np.arange(x[0], x[-1], 0.1)
ys50_new = S50_BSpline(xs50_new)
for i in range(len(ys50_new)):
    if ys50_new[i] <= 0:
        ys50_new[i] = 0

#End_50
E50BSpline = interpolate.make_interp_spline(x, end_50_l)
xe50_new = np.arange(x[0], x[-1], 0.1)
ye50_new = E50BSpline(xe50_new)
for i in range(len(ye50_new)):
    if ye50_new[i] <= 0:
        ye50_new[i] = 0

#Start_75
S75BSpline = interpolate.make_interp_spline(x, start_75_l)
xs75_new = np.arange(x[0], x[-1], 0.1)
ys75_new = S75BSpline(xs75_new)
for i in range(len(ys75_new)):
    if ys75_new[i] <= 0:
        ys75_new[i] = 0

#End_75
E75BSpline = interpolate.make_interp_spline(x, end_75_l)
xe75_new = np.arange(x[0], x[-1], 0.1)
ye75_new = E75BSpline(xe75_new)
for i in range(len(ye75_new)):
    if ye75_new[i] <= 0:
        ye75_new[i] = 0

#Start_95
S95BSpline = interpolate.make_interp_spline(x, start_95_l)
xs95_new = np.arange(x[0], x[-1], 0.1)
ys95_new = S95BSpline(xs95_new)
for i in range(len(ys95_new)):
    if ys95_new[i] <= 0:
        ys95_new[i] = 0

#End_95
E95BSpline = interpolate.make_interp_spline(x, end_95_l)
xe95_new = np.arange(x[0], x[-1], 0.1)
ye95_new = E95BSpline(xe95_new)
for i in range(len(ye95_new)):
    if ye95_new[i] <= 0:
        ye95_new[i] = 0


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x_ticks = np.arange(0, 11, 1)

ax.set_xticks(x_ticks)



ax.grid()
ax.plot(xm_new, ym_new, 'r-', label = "Mean")
ax.plot(xs50_new,ys50_new, 'y-', label = "50% R", markersize=4)
ax.plot(xe50_new,ye50_new, 'y-', label = "50% L", markersize=4)
ax.plot(xs75_new,ys75_new, 'g-', label = "75% R", markersize=4)
ax.plot(xe75_new,ye75_new, 'g-', label = "75% L", markersize=4)
ax.plot(xs95_new,ys95_new, 'b-', label = "95% R", markersize=4)
ax.plot(xe95_new,ye95_new, 'b-', label = "95% L", markersize=4)
ax.legend()
plt.show()
"""
Working version - DONT TOUCH!!!
plt.plot(xm_new, ym_new, x,m_list, 'r-', label = "Mean")
plt.plot(x,start_50_l, 'y-', label = "50% R", markersize=4)
plt.plot(x,end_50_l, 'y-', label = "50% L", markersize=4)
plt.plot(x,start_75_l, 'g-', label = "75% R", markersize=4)
plt.plot(x,end_75_l, 'g-', label = "75% L", markersize=4)
plt.plot(x,start_95_l, 'b-', label = "95% R", markersize=4)
plt.plot(x,end_95_l, 'b-', label = "95% L", markersize=4)
plt.legend()
plt.show()
"""



