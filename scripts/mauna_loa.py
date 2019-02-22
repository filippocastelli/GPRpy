import numpy as np
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt

import sys 
sys.path.append('..')
from GPRpy import GPRpy
#%%
def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs
#%%
X_full, y_full = load_mauna_loa_atmospheric_co2()
cutoff = 100

X = X_full[:-cutoff]
y = y_full[:-cutoff]

X_test = X_full[-cutoff :]
y_test = y_full[-cutoff :]

X_ = np.linspace(X_full.min(), X_full.max() + 20, 1000)
plt.figure()

plt.title("Dati concentrazione CO2 Mauna Loa")
ax = plt.gca()
dati_train,= ax.plot(X, y)
dati_test,= ax.plot(X_test, y_test)
plt.legend([dati_train, dati_test], ["dati di training", "dati di test"])
plt.xlabel("anno")
plt.ylabel("concentrazione CO2 [ppmv]")
plt.savefig("maunaloa_co2_dataoverview.png")
#%%
# MODEL SETTING
model_params = {
'RBF_const': 51,
'RBF_length':87,
'RBFperiodic_const':2.4,
'RBFperiodic_length':100,
'PERIODIC_length':1.3,
'RADQUAD_const':0.66, 
'RADQUAD_length':1.2,
'RADQUAD_shape':0.78,
'RBFnoise_length':0.15,
'RBFnoise_const':0.18}

gpr = GPRpy(x = X,
                 y = y,
                 x_guess = X_,
                 kernel = GPRpy.mauna_loa_example_kernel2,
                 kernel_params = model_params,
                 normalize_y = True,
                 R = 0.0361)
gpr.predict()
#%%
gpr.plot(title = "Gaussian Process Regression, dati Mauna Loa, dettaglio",
         axlabels = ["anno [yr]", "concentrazione CO2 [ppmv]"],
         save = "mauna_loa_regression",
         return_ax = True,
         figsize = [25,10])
#%%
ax = gpr.plot(title = "Gaussian Process Regression, dati Mauna Loa, dettaglio",
         axlabels = ["anno [yr]", "concentrazione CO2 [ppmv]"],
         return_ax = True,
         figsize = [20,10],
         axlims = [(1987, 2011), (342, 382)])
ax.scatter(X_test, y_test, label = "dati test")
ax.legend()
plt.savefig("mauna_loa_test_data")