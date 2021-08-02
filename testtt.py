import numpy as np
from numpy.core.defchararray import array
import numpy as np, scipy.stats as st
from numpy.lib.function_base import trapz
from scipy import stats
from scipy.stats import sem, t
from scipy import mean
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate 
from sklearn.neighbors import KernelDensity
np.set_printoptions(suppress=True)

df = pd.read_excel ('book.xlsx')
df_list = df.to_numpy()
tr_dflist = df_list.transpose()

tr_dflist = tr_dflist.tolist()

col_list = tr_dflist[1]

col_list.sort()



x1 = np.array(col_list).reshape(-1,1)
kde1 = KernelDensity(bandwidth=1, kernel='gaussian').fit(x1)


padding = 3
numOfPoints = 8000
r = np.linspace(x1.min()- padding, x1.max()+ padding, numOfPoints) 
area = integrate.trapz(np.exp(kde1.score_samples(r.reshape(-1,1))),r )


ll = [76767,0.8,3,6,2,8,4,26,7,315,78]

ll = np.array(ll).reshape(-1,1)
print(ll)