import numpy as np
import pyDOE as pyDOE
import pickle

# number of parameters and samples
n_params = 11
n_samples = 21 #199980 #Number of training samples -- this should be a multiple of num_subfile below because this code will distribute the axionCAMB runs across the CPUs that are available

# parameter ranges
obh2 =      np.linspace(0.022381, 0.022381, n_samples) #0.0221, 0.0227, n_samples) #0.021, 0.024, n_samples) #0.01933, 0.02533, n_samples) #0.0222, 0.0229, n_samples) #0.017, 0.027, n_samples)
H_0 =         np.linspace(67.24, 67.24, n_samples) #66., 69., n_samples) #62., 75., n_samples) #39.99, 100.01, n_samples) #64., 68., n_samples) #55.0,    82.0,    n_samples) 
ns =        np.linspace(0.9684, 0.9684, n_samples) #0.956, 0.981, n_samples) #0.92, 1.01, n_samples) #0.8, 1.2, n_samples) #0.95, 0.98, n_samples) #0.86, 1.07,    n_samples)
As =      np.linspace(2.107e-9, 2.107e-9, n_samples) #2.09e-9, 2.15e-9, n_samples) #1.5e-9, 2.5e-9, n_samples) #1.2182493960703474e-09, 3.311545195869231e-09, n_samples) #2.06e-9, 2.2e-9, n_samples) #1.3e-9, 2.8e-9 #5e-10,    2.6e-9,    n_samples)
tau_reio = np.linspace(0.0559, 0.0559, n_samples) #0.0525, 0.0645, n_samples) #0.02, 0.09, n_samples) #0.02, 0.12, n_samples) #0.043, 0.069, n_samples) #0.01, 0.1 #0.02, 0.12,    n_samples)
z = np.linspace(2.51,5.0, n_samples) #For the given range of z here, you will end up with another set of samples mirrored around the minimum value, i.e. from 0 to 2.5
ma = np.linspace(-24., -24., n_samples) #np.array([-27., -26., -25., -24., -23.]) #np.concatenate((np.linspace(-28., -21., n_samples-1), np.array([-17.,]))) #-27, -23, n_samples)
sum_omega = np.linspace(0.12009, 0.12009, n_samples) #0.116, 0.122, n_samples) #0.1, 0.14, n_samples) #0.08, 0.2, n_samples) #0.12, 0.13, n_samples) #0.09, 0.15, n_samples) # sum of omaxh2 and omlambda in dark-energy region #sum of omaxh2 and omch2 in dark-matter region
#0.124, 0.565
f_ax = np.linspace(1.e-8, 1., n_samples) #1
#These are experimental halo model parameters -- leave them fixed to zero
gamma_1 = np.linspace(0., 0., n_samples) #45., n_samples) #np.linspace(5., 45., n_samples)
gamma_2 = np.linspace(0., 0., n_samples) #0.37, n_samples) #np.linspace(-0.37, -0.23, n_samples)

# LHS grid
AllParams = np.vstack([obh2, H_0, ns, As, tau_reio, z, ma, sum_omega, f_ax, gamma_1, gamma_2])
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)

AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving
params = {'omega_b': AllCombinations[:, 0],
          'omega_ax': AllCombinations[:, 7] * AllCombinations[:, 8],
          'H_0': AllCombinations[:, 1],
          'n_s': AllCombinations[:, 2],
          'A_s': AllCombinations[:, 3],
          'tau_reio': AllCombinations[:, 4],
          'z': AllCombinations[:, 5],
          'ma':10**(AllCombinations[:, 6]),
          'omega_cdm': AllCombinations[:, 7] * (1. - AllCombinations[:, 8]), #'omega_cdm': ((AllCombinations[:, 1] / 100.)**2.) - AllCombinations[:, 7] - AllCombinations[:, 0] - 0.0006442, #CDM is total - (ax + Lambda) - baryons - neutrinos
          'gamma_1':AllCombinations[:, 9],
          'gamma_2':AllCombinations[:, 10]
           }

print(params)

data_pkl = '/home/keir/keir/LH_ACT_DR6_TTTEEEPP_21_test_axion.pkl' #the .pkl that stores all input parameters
f = open(data_pkl, 'wb')
pickle.dump(params, f)
f.close()
num_subfile = 21 # this number should be in consistent with number_cores variable in the 9parameters_data_collection_mp.py file
num_samples_per_subfile = int(n_samples/num_subfile)
for i in range(num_subfile):
    start = int(i*num_samples_per_subfile)
    print(start, start+num_samples_per_subfile)
    params_1 = {'omega_b': params['omega_b'][start:start+num_samples_per_subfile],
          'omega_cdm': params['omega_cdm'][start:start+num_samples_per_subfile],
          'H_0': params['H_0'][start:start+num_samples_per_subfile],
          'n_s': params['n_s'][start:start+num_samples_per_subfile],
          'A_s': params['A_s'][start:start+num_samples_per_subfile],
          'tau_reio': params['tau_reio'][start:start+num_samples_per_subfile],
          'z': params['z'][start:start+num_samples_per_subfile],
          'ma': params['ma'][start:start+num_samples_per_subfile],
          'omega_ax': params['omega_ax'][start:start+num_samples_per_subfile],
          'gamma_1': params['gamma_1'][start:start+num_samples_per_subfile],
          'gamma_2': params['gamma_2'][start:start+num_samples_per_subfile]
           }
    data_pkl = '/home/keir/keir/LH_ACT_DR6_TTTEEEPP_21_test_axion_' +str(i) +'.pkl'
    f = open(data_pkl, 'wb')
    pickle.dump(params_1, f)
    f.close()

