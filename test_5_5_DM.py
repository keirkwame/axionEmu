# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import os

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

#!pip install cosmopower

# from google.colab import drive
# drive.mount('/content/drive')

model_parameters = ['h',
                    'z',
                    'omega_b',
                    'n_s',
                    'ln10A_s',
                    'omega_cdm',
                    'log10ma',
                    'omega_ax']

#list_index = np.linspace(0,79,80)
##list_index = np.linspace(80, 239, 160) #DE
##list_index = np.delete(list_index, 167 - 80) #DE
list_index = np.linspace(0, 59, 60)
import pickle
collection_list = []
for i in list_index:
    if i>-1:
        i = int(i)
        #f = open('/home/anran/axionCAMB/test_data_collect_mg1_9params_5e5_mp_test_'+str(i)+'.pkl', 'rb')
        f = open('/home/keir/Software/axionEmu/data_collection_axion/test_data_collect_mg1_HMcode_NL_200k_fixh2_' +str(i)+'.pkl', 'rb')
        collection = pickle.load(f)
        f.close()
        collection_list.append(collection)

for i in list_index:
    if i>-1:
        i = int(i)
        #f = open('/home/anran/axionCAMB/test_data_collect_mg2_9params_5e5_mp_test_'+str(i)+'.pkl', 'rb')
        f = open('/home/keir/Software/axionEmu/data_collection_axion/test_data_collect_mg2_HMcode_NL_200k_fixh2_' +str(i)+'.pkl', 'rb')
        collection = pickle.load(f)
        f.close()
        collection_list.append(collection)

print(len(collection_list), collection_list[0].keys())

print("Open CDM (Pk) file")
f_ = open("./test_data_collect_mg_cdm.pkl", 'rb')
c_ = pickle.load(f_)
f_.close()
pk_cdm = c_['matter_mg'][0] #[:244] #576
pk_k = c_['k_index'][0] #[:244]

import random
random.shuffle(collection_list)

parameters_list = {}
for key in collection['params']:
    parameters_list[key] = np.array([])

k_index_list = []
mg_list = []

for i in range(len(collection_list)):
    para = collection_list[i]['params']
    k_index_list.extend(collection_list[i]['k_index'])
    mg_list.extend(collection_list[i]['matter_mg'])
    for key in para:
      parameters_list[key] = np.concatenate((parameters_list[key], para[key]))

#print(mg_list)

'''for i in range(len(mg_list)):
    mg_list[i] = mg_list[i][:244]
min = 244
k_min_max = 0.
k_max_min = 1.e+4
for i in range(len(k_index_list)):
    if len(k_index_list[i]) < min:
        min = len(k_index_list[i])
    if k_index_list[i][0] > k_min_max:
        k_min_max = k_index_list[i][0]
    if k_index_list[i][-1] < k_max_min:
        k_max_min = k_index_list[i][-1]
print('The minimum length of k_index is ', min)
print('The maximum k_min is', k_min_max)
print('The minimum k_max is', k_max_min)
for i in range(len(k_index_list)):
    k_index_list[i] = k_index_list[i][:244]
'''
#Interpolate to same k values
k_min_max = 1.e-3 #1.7301e-05
k_max_min = 1.e+3
k_new = np.logspace(np.log10(k_min_max), np.log10(10.), num=400)
spectra_ = np.zeros((len(mg_list), k_new.shape[0]))
for i, mg in enumerate(mg_list):
    spectra_[i, :] = np.interp(np.log10(k_new), np.log10(k_index_list[i]), np.log10(mg))
pk_cdm_interp = 10. ** np.interp(np.log10(k_new), np.log10(pk_k), np.log10(pk_cdm))

#Take log of power spectra
#spectra_= np.array(mg_list)
#spectra_ = np.log10(spectra_)
print(spectra_.shape)

#Remove nan spectra
spectra_bool = np.array(np.sum(~np.isfinite(spectra_), axis=1), dtype=bool) #True if not finite element
print(np.sum(spectra_bool))
spectra_ = spectra_[~spectra_bool, :] #True if finite element
for key in parameters_list.keys():
        parameters_list[key] = parameters_list[key][~spectra_bool]
        print(parameters_list[key].shape)
print(spectra_.shape)

#Remove high-axion-fraction spectra
spectra_bool_axion = parameters_list['omega_ax'] < 0.07
print(np.sum(spectra_bool_axion))
spectra_ = spectra_[spectra_bool_axion, :]
for key in parameters_list.keys():
        parameters_list[key] = parameters_list[key][spectra_bool_axion]
        print(parameters_list[key].shape)
print(spectra_.shape)

cut_off = 193000 #288000
import copy
training_parameters_ = copy.deepcopy(parameters_list)
h_0 = training_parameters_['H_0'][:cut_off]/100
ln10_10A_s = np.log(training_parameters_['A_s'][:cut_off]*10**10)
ma_mass = np.array(training_parameters_['ma'][:cut_off]*10**32,dtype='float64')
ma_mass = np.log10(ma_mass)
training_parameters = dict()
training_parameters['omega_b'] = training_parameters_['omega_b'][:cut_off]
training_parameters['omega_cdm'] = training_parameters_['omega_cdm'][:cut_off]
training_parameters['h'] = h_0
training_parameters['z'] = training_parameters_['z'][:cut_off]
training_parameters['n_s'] = training_parameters_['n_s'][:cut_off]
training_parameters['ln10A_s'] = ln10_10A_s
training_parameters['log10ma'] = ma_mass
training_parameters['omega_ax'] = training_parameters_['omega_ax'][:cut_off]
ell_range = np.linspace(2,6000,5999)

test_parameters_ = copy.deepcopy(parameters_list)
h_0 = test_parameters_['H_0'][cut_off:]/100
ln10_10A_s = np.log(test_parameters_['A_s'][cut_off:]*10**10)
ma_mass = np.array(test_parameters_['ma'][cut_off:]*10**32, dtype = 'float64')
ma_mass = np.log10(ma_mass)
test_parameters = dict()
test_parameters['omega_b'] = test_parameters_['omega_b'][cut_off:]
test_parameters['omega_cdm'] = test_parameters_['omega_cdm'][cut_off:]
test_parameters['h'] = h_0
test_parameters['z'] = test_parameters_['z'][cut_off:]
test_parameters['n_s'] = test_parameters_['n_s'][cut_off:]
test_parameters['ln10A_s'] = ln10_10A_s
test_parameters['omega_ax'] = test_parameters_['omega_ax'][cut_off:]
test_parameters['log10ma'] = ma_mass

training__spectra = spectra_[:cut_off,:]
testing_spectra = spectra_[cut_off:,:]
print('number of training set: ', training__spectra.shape)
print('number of test set: ', testing_spectra.shape)

om0 = 0.3158
oml = 1. - 0.3158

def omz(z):
    """Fraction of cold dark matter in cold dark matter and dark energy density."""
    return om0 * ((1. + z) ** 3.) / ((om0 * ((1. + z) ** 3.)) + oml)

def growth_factor_D(z):
    """Cosmological growth factor D(z)."""
    return 5. * omz(z) / 2. / (1. + z) / ((omz(z) ** (4. / 7.)) - ((omz(z) ** 2.) / 140.) + (209. * omz(z) / 140) + (1. / 70.))

from cosmopower import cosmopower_NN

#Train new emulators
cp_nn = cosmopower_NN(parameters=model_parameters,
                      modes=k_new, #k_index_list[0],
                      n_hidden = [64, 512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

print('Number of nan elements =', np.sum(np.isnan(training__spectra)))

with tf.device('/device:GPU:0'): # ensures we are running on a GPU
    # train
    cp_nn.train(training_parameters=training_parameters,
                training_features=training__spectra,
                filename_saved_model='Pk_cp_NN_NL_t2',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-6],
                batch_sizes=[1024,1024,1024,1024,1024,1024],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [200,200,200,200,200,200],
                max_epochs = [1000,1000,1000,1000,1000, 1000],
                )

cp_nn = cosmopower_NN(restore=True,
                      restore_filename='Pk_cp_NN_NL_t2',
                      )
testing_spectra_ex = []
testing_parameters_ex = dict()
testing_parameters_ex['omega_b'] = []
testing_parameters_ex['omega_cdm'] = []
testing_parameters_ex['h'] = []
testing_parameters_ex['z'] =[]
testing_parameters_ex['n_s'] = []
testing_parameters_ex['log10ma'] = []
testing_parameters_ex['ln10A_s'] = []
testing_parameters_ex['omega_ax'] = []
neg_ = 0
for k in range(len(testing_spectra)):
    #print(test_parameters['z'][k], (growth_factor_D(2.5) ** 2.) / (growth_factor_D(test_parameters['z'][k]) ** 2.))
    if np.min(10**(testing_spectra[k]) * (growth_factor_D(2.5) ** 2.) / (growth_factor_D(test_parameters['z'][k]) ** 2.) / pk_cdm_interp)>=0.01:
        if test_parameters['omega_ax'][k] < 0.07:
            testing_spectra_ex.append(testing_spectra[k])
            testing_parameters_ex['omega_b'].append(test_parameters['omega_b'][k])
            testing_parameters_ex['omega_cdm'].append(test_parameters['omega_cdm'][k])
            testing_parameters_ex['h'].append(test_parameters['h'][k])
            testing_parameters_ex['z'].append(test_parameters['z'][k])
            testing_parameters_ex['n_s'].append(test_parameters['n_s'][k])
            testing_parameters_ex['ln10A_s'].append(test_parameters['ln10A_s'][k])
            testing_parameters_ex['log10ma'].append(test_parameters['log10ma'][k])
            testing_parameters_ex['omega_ax'].append(test_parameters['omega_ax'][k])
        else:
            neg_ += 1
    else:
        neg_ += 1
testing_spectra_ex = np.array(testing_spectra_ex)
print("The number of data points excludes is ", neg_)
predicted_testing_spectra = cp_nn.predictions_np(testing_parameters_ex)
from matplotlib import gridspec
diff = np.abs(10**(predicted_testing_spectra) - 10**(testing_spectra_ex))/(10**(testing_spectra_ex))
# Compute percentiles
percentiles = np.zeros((4, diff.shape[1]))

percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)

plt.figure(figsize=(12, 9))
plt.fill_between(k_new, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(k_new, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(k_new, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)
plt.xscale("log")
# plt.ylim(0, 0.2)

plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k)^{emulated} - P(k)^{true}|} {P(k)^{true}}$', fontsize=30)
plt.xlabel(r'$\k$',  fontsize=30)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.LogLocator(base=10,numticks=6))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)
plt.tight_layout()
plt.savefig('./accuracy_emu_Pk_NL_notrain_ax0.07_exclude_1pc_t2.pdf')

plt.figure(figsize=(12, 9))
z_plot = np.linspace(0., 5., num=100)
growth_ratio = np.array([growth_factor_D(z) for z in z_plot])
plt.plot(z_plot, growth_ratio * (1. + z_plot) / growth_factor_D(0.))
plt.xlabel(r'$z$')
plt.ylabel(r'$D(z) / a(z) / D(z = 0)$')
plt.savefig('./D_z_norm_t2.pdf')
