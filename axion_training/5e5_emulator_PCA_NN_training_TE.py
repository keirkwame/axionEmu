# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt
import os

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

#names of input parameters
model_parameters = ['h', 
                    'tau_reio', 
                    'omega_b', 
                    'n_s', 
                    'ln10A_s', 
                    'omega_cdm',
                    'log10ma',
                    'omega_ax' 
                    ]
list_index = np.linspace(0,239,240)
list_index = np.delete(list_index, 167)
import pickle
collection_list = []
for i in list_index:
    i = int(i)
    f = open('./test_data_collect_9params_5e5_mp_test_'+str(i)+'.pkl', 'rb')
    collection = pickle.load(f)
    f.close()
    print(len(collection['C_tt']))
    collection_list.append(collection)

import random
random.shuffle(collection_list)

parameters_list = {}
for key in collection['params']:
    parameters_list[key] = np.array([])

# collection['params'].keys()

C_tt_list = []
C_ee_list = []
C_bb_list = []
C_te_list = []
C_phi_list =[]

for i in range(239):
    para = collection_list[i]['params']
    C_tt_list.extend(collection_list[i]['C_tt'])
    C_ee_list.extend(collection_list[i]['C_ee'])
    C_bb_list.extend(collection_list[i]['C_bb'])
    C_te_list.extend(collection_list[i]['C_te'])
    C_phi_list.extend(collection_list[i]['C_phi'])
    for key in para:
      parameters_list[key] = np.concatenate((parameters_list[key], para[key]))

cut_off = 360000
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
training_parameters['tau_reio'] = training_parameters_['tau_reio'][:cut_off]
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
test_parameters['tau_reio'] = test_parameters_['tau_reio'][cut_off:]
test_parameters['n_s'] = test_parameters_['n_s'][cut_off:]
test_parameters['ln10A_s'] = ln10_10A_s
test_parameters['omega_ax'] = test_parameters_['omega_ax'][cut_off:]
test_parameters['log10ma'] = ma_mass

spectra_= np.array(C_te_list)/(7.4311*10**(12))
print(spectra_.shape)
spectra_ = spectra_[:,:5999]/(ell_range*(ell_range+1)/(2.*np.pi))
training__spectra = spectra_[:cut_off,:]
testing_spectra = spectra_[cut_off:,:]
print('number of training set: ', training__spectra.shape)
print('number of test set: ', testing_spectra.shape)

# Generate npz files
np.savez_compressed('./training_params_TE_5e5_2.npz', omega_b = training_parameters['omega_b'], 
         omega_cdm = training_parameters['omega_cdm'],
         h = training_parameters['h'],
         tau_reio = training_parameters['tau_reio'],
         n_s = training_parameters['n_s'],
         ln10A_s = training_parameters['ln10A_s'],
         log10ma = training_parameters['log10ma'],
         omega_ax = training_parameters['omega_ax'])

npzfile = np.load('./training_params_TE_5e5_2.npz')
print(npzfile.files)

np.savez_compressed('./training_feature_TE_5e5_2.npz', mode = np.linspace(2,6000,5999), 
         features = training__spectra)

npzfile = np.load('./training_feature_TE_5e5_2.npz')
print(npzfile.files)

n_pcas = 64
from cosmopower import cosmopower_PCA

cp_pca = cosmopower_PCA(parameters=model_parameters,
                        modes=ell_range,
                        n_pcas=n_pcas,
                        parameters_filenames=['./training_params_TE_5e5_2'],
                        features_filenames=['./training_feature_TE_5e5_2'],
                        verbose=True, # useful to follow the various steps
                        )

cp_pca.transform_and_stack_training_data()

from cosmopower import cosmopower_PCAplusNN

cp_pca_nn = cosmopower_PCAplusNN(cp_pca=cp_pca,
                                 n_hidden=[64, 512,512,512,512], # 4 hidden layers, each with 512 nodes
                                 verbose=True, # useful to understand the different steps in initialisation and training
                                 )

with tf.device('/device:GPU:0'): # ensures we are running on a GPU
    # train
    cp_pca_nn.train(filename_saved_model='TE_cp_PCAplusNN_5e5_t2',
                    # cooling schedule
                    validation_split=0.1,
                    learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                    batch_sizes=[1024, 1024, 1024, 1024, 1024],
                    gradient_accumulation_steps = [1, 1, 1, 1, 1],
                    # early stopping set up
                    patience_values = [100,100,100,100,100],
                    max_epochs = [1000,1000,1000,1000,1000],
                    )

import cosmopower
from cosmopower import cosmopower_PCAplusNN
cp_pca_nn = cosmopower_PCAplusNN(restore=True,
                                 restore_filename='TE_cp_PCAplusNN_5e5_t2'
                                 )

predicted_testing_spectra = cp_pca_nn.predictions_np(test_parameters)

from matplotlib import gridspec
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50,10))
for i in range(3):
    pred = predicted_testing_spectra[i]*ell_range*(ell_range+1)/(2.*np.pi)
    true = testing_spectra[i]*ell_range*(ell_range+1)/(2.*np.pi)
    ax[i].semilogx(ell_range, true, 'blue', label = 'Original')
    ax[i].semilogx(ell_range, pred, 'red', label = 'NN reconstructed', linestyle='--')
    ax[i].set_xlabel('$\ell$', fontsize='x-large')
    ax[i].set_ylabel('$\\frac{\ell(\ell+1)}{2 \pi} C_\ell$', fontsize='x-large')
    ax[i].legend(fontsize=15)
plt.savefig('examples_reconstruction_TE_PCANN_5e5_2.pdf')
# load noise models from the SO noise repo
noise_levels_load_T = np.loadtxt('./so_noise_models/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt')
noise_levels_load_E = np.loadtxt('./so_noise_models/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_E.txt')
conv_factor = (2.7255e6)**2


ells = noise_levels_load_T[:, 0]
SO_TT_noise_NN = noise_levels_load_T[:, 1][:6001-40] / conv_factor
SO_EE_noise_NN = noise_levels_load_E[:, 1][:6001-40] / conv_factor
new_ells = ells[:6001-40]
ell_range = np.linspace(2,6000,5999)
T_spectra_= np.array(C_tt_list)/(7.4311*10**(12))
E_spectra_= np.array(C_ee_list)/(7.4311*10**(12))
T_spectra_ = T_spectra_[:,:5999]/(ell_range*(ell_range+1)/(2.*np.pi))
E_spectra_ = E_spectra_[:,:5999]/(ell_range*(ell_range+1)/(2.*np.pi))
T_true = T_spectra_[cut_off:,:]
E_true = E_spectra_[cut_off:,:]

f_sky = 0.4
prefac = np.sqrt(2/(f_sky*(2*new_ells+1)))
denominator = prefac*np.sqrt(np.multiply(testing_spectra[:, 38:],testing_spectra[:, 38:])+np.multiply(T_true[:,38:] + SO_TT_noise_NN, E_true[:,38:] + SO_EE_noise_NN))  # use all of them
diff = np.abs((predicted_testing_spectra[:, 38:] - testing_spectra[:, 38:])/(denominator))

# Compute percentiles
percentiles = np.zeros((4, diff.shape[1]))

percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)

plt.figure(figsize=(12, 9))
plt.fill_between(new_ells, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(new_ells, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(new_ells, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)

# plt.ylim(0, 0.2)

plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| C_{\ell, \rm{emulated}}^{\rm{TE}} - C_{\ell, \rm{true}}^{\rm{TE}}|} {\sigma_{\ell, \rm{CMB}}^{\rm{TE}}}$', fontsize=50)
plt.xlabel(r'$\ell$',  fontsize=50)

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)
plt.tight_layout()
plt.savefig('./accuracy_emu_TE_wide_PCANN_5e5_2.pdf')
diff_=np.sum(diff,axis = 1)
print('diff_.shape is ', diff_.shape)
sort_index = np.argsort(diff_)
bad_params = dict()
for key in test_parameters:
    bad_params[key] = []
for i in sort_index[len(sort_index)-100:]:
    for key in test_parameters:
        bad_params[key].append(test_parameters[key][i])
fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(50,10))
for i in range(8):
    pa =list(test_parameters.keys())[i]
    h = bad_params[pa]
    ax[i].plot(h, 'o')
    ax[i].set_ylabel(pa, fontsize='x-large')
plt.savefig('bad_params_TE_PCANN_5e5_2.pdf')
