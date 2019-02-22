import sys 
sys.path.append('..')
from GPRpy import GPRpy

import numpy as np
import matplotlib.pylab as plt

def f1(x):
    return np.sin(x.T)
#%%
N = 50     # numero punti training
n = 1000   # numero punti test
s = 0.1   # noise variance

rng = np.random.RandomState(2)
x = np.squeeze(rng.uniform(-5, 5, size = (N,1)))
x_guess = np.linspace(-5, 5, n)
y = f1(x) + s*np.random.randn(N)
#%%
# PLOT MISURE
plt.figure()
plt.title("Misure")
ax = plt.gca()
cosine, = ax.plot(x_guess, f1(x_guess))
measures = plt.scatter(x,y, c = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.savefig('misure.png', bbox_inches='tight')

#%%
gaus = GPRpy(x,y,
                x_guess,
                kernel=GPRpy.gaussian_kernel,
                kernel_params = {'const': 10,
                                 'length': 5},
                R =s)
#%%
preds = gaus.predict()
gaus.plot(axlabels = ["x", "y"],title = 'Before Optimization', save = "test_figure")
#%%
optimizer_params = {'const': np.log(1),
                    'length': np.log(10)}
min_results = gaus.optimizer(mode='CG',param_x0 = optimizer_params)
#%%
min_results = gaus.optimizer(mode='Newton-CG',param_x0 = optimizer_params)
#%%
optimized_params, logp = gaus.optimizer(mode='Nelder-Mead',param_x0 = optimizer_params)
#%%
optimizer_ranges = {'const': (0,10),
                    'length': (0,10)}
optimized_params, logp, grid, grid_values = gaus.optimizer(ranges_dict = optimizer_ranges,
                                                           mode = "brute",
                                                           Ns = 500,
                                                           output_grid = True)
#%%
gaus.update_params(optimized_params)
preds2 = gaus.predict()
gaus.plot(title = "after optim",axlabels = ["x", "y"], save = "optim")

