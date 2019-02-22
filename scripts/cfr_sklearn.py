import sys 
sys.path.append('..')
from GPRpy import GPRpy

import numpy as np
import matplotlib.pylab as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def f(x):
    """The function to predict."""
    return x * np.sin(x)
#%%
# REGRESSION CASE
N = 5     # numero punti training
n = 1000  # numero punti test
s = 0.001   # noise variance

rng = np.random.RandomState(5)
X = rng.uniform(0, 5, size = (N,1))
y = f(X).ravel() + s*np.random.randn(N)
x = np.atleast_2d(np.linspace(0, 5, n)).T

#x1 = rng.uniform(0, 5, size = (N,1))
#%%
kernel_sklearn = C(4.71, (1e-3, 1e3)) * RBF(1.68, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel_sklearn,
                              n_restarts_optimizer=0,
                              normalize_y = False)
gp.fit(X, y)
y_pred, sigma = gp.predict(x, return_std=True)


plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - sigma,
                        (y_pred + sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
ax = plt.gca()

sklearn_params = gp.kernel_.get_params()

sk_const = np.sqrt(sklearn_params['k1__constant_value'])
sk_length_scale = sklearn_params['k2__length_scale']

params = {'const': np.exp(sk_const),
          'length': np.exp(sk_length_scale)}

gaus = GPRpy(x = X, y =y,
                  x_guess = x,
                  kernel = GPRpy.gaussian_kernel,
                  kernel_params = params,
                  R = 0.,
                  normalize_y = False)

gaus_predict, gaus_var, gaus_logp = gaus.predict()
gaus.plot(ax = ax)
ax.set_ylim([-6,2])
