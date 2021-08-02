import numpy as np, scipy.stats as st
from numpy.lib.arraypad import pad
from scipy import stats
from scipy.stats import sem, t
import pandas as pd
from matplotlib import pyplot as plt
import scipy.integrate as integrate 
from sklearn.neighbors import KernelDensity
np.set_printoptions(suppress=True)


df = pd.read_excel ('book.xlsx')
df_list = df.to_numpy()
tr_dflist = df_list.transpose()

tr_dflist = tr_dflist.tolist()

col_list = tr_dflist[1]
col_list = np.array(col_list)
kde_col_list = stats.gaussian_kde(col_list)

col_list.sort()

area_1 = kde_col_list.integrate_box_1d(col_list[0], col_list[-1])


####Conf interval calculating
a = np.array(col_list)
inter = st.t.interval(0.5, len(a)-1, loc=np.mean(a), scale=st.sem(a))


confidence_50 = 0.5
confidence_75 = 0.75
confidence_95 = 0.95


n = len(col_list)
m = np.mean(col_list)
std_err = sem(col_list)


"""
np.mean - Compute standard error of the mean.
            Calculate the standard error of the mean (or standard error of measurement) of the values in the input array.

sem-Compute standard error of the mean.
    Calculate the standard error of the mean (or standard error of measurement) of the values in the input array.

t.ppf - Percent point function (inverse of cdf — percentiles, cdf - Cumulative distribution function.).

Endpoints of the range that contains alpha percent of the distribution
"""
h_50 = std_err * t.ppf((1 + confidence_50) / 2, n - 1)
h_75 = std_err * t.ppf((1 + confidence_75) / 2, n - 1)
h_95 = std_err * t.ppf((1 + confidence_95) / 2, n - 1)


start_50 = m - h_50
end_50 = m + h_50
# print(f"start_50: {start_50},\nend_50: {end_50}")


start_75 = m - h_75
end_75 = m + h_75

start_95 = m - h_95
end_95 = m + h_95


conf_50 = []
conf_50.append(start_50)
conf_50.append(end_50)


conf_75 = []
conf_75.append(start_75)
conf_75.append(end_75)


conf_95 = []
conf_95.append(start_95)
conf_95.append(end_95)



###Area
x = np.array(col_list).reshape(-1,1)
kde1 = KernelDensity(bandwidth=1, kernel='gaussian').fit(x)

padding = 3
numOfPoints = 2000
r = np.linspace(x.min()- padding, x.max()+ padding, numOfPoints) 
log_dens = kde1.score_samples(r.reshape(-1,1))
expon = np.exp(log_dens)
area = integrate.trapz(expon,r)
plt.plot(r, expon, 'yo')

print(len(log_dens), f'\n{len(r.reshape(-1,1))}')

r_50 = []

for i in range(len(expon)):
    # print(expon[i])
    if r[i] >= start_50 and r[i] <= end_50:
        r_50.append(r[i])

r_50 = np.array(r_50)

log_dens_50 = kde1.score_samples(r_50.reshape(-1,1))
exp_50 = np.exp(log_dens_50)

print(len(log_dens_50), f'\n{len(r_50.reshape(-1,1))}')

# area_50 = integrate.trapz(exp_50, r_50)
# for i in range(len(log_dens)):
#     print(log_dens[i])



####


y = np.arange(len(col_list))

fig, ax = plt.subplots()

ax.plot(col_list,y, '-')
ax.plot([m, m], [0, 120], '-r', label = f"E: {m}")

plt.plot([conf_50[0], conf_50[0]],[0,120], 'y', label = f"50%: {start_50}, {end_50}")
plt.plot([conf_50[1], conf_50[1]],[0,120], 'y')

plt.plot([conf_75[0], conf_75[0]],[0,120], 'm', label = f"75%: {start_75}, {end_75}")
plt.plot([conf_75[1], conf_75[1]],[0,120], 'm')

plt.plot([conf_95[0], conf_95[0]],[0,120], 'b', label = f"95%: {start_95}, {end_95}")
plt.plot([conf_95[1], conf_95[1]],[0,120], 'b')
ax.legend()
plt.show()


# df = pd.DataFrame(col_list)
# df_50 = pd.DataFrame(conf_50)
# df_75 = pd.DataFrame(conf_75)
# df_95 = pd.DataFrame(conf_95)
# with pd.ExcelWriter('.output.xlsx') as writer:
#     df.to_excel(writer, sheet_name='Data')
#     df_50.to_excel(writer, sheet_name='50%')
#     df_75.to_excel(writer, sheet_name='75%')
#     df_95.to_excel(writer, sheet_name='95%')




