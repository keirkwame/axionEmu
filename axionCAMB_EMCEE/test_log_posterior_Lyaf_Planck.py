#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from test_log_likelihood import log_likelihood_Lyaf, log_likelihood
#from test_log_prior import log_prior
#import emcee
import time
import os
from multiprocessing import Pool
import pickle
import pypolychord as pp
import pypolychord.priors as ppp
import pypolychord.settings as pps
os.environ["OMP_NUM_THREADS"] = "1"


def log_posterior(x, parameters_and_priors):
    """Return the value of the log posterior given x = theta, a list contains parameters' values 
    and parameters_and_priors, a dict contains information for calculating log_prior.
    ## parameters_and_priors order matters!! ##
    x = ['omega_b',
           'omega_cdm':,
           'H_0',
           'tau_reio',
           'n_s',
           'ln10^10A_s',
        'A_plank'    ]
    parameters_and_priors = 
    {'omega_b':      [0.001, 0.04, 'uniform'],
                          'omega_cdm':    [0.005, 0.99,  'uniform'],
                          'H_0':            [20.0,   80.0,   'uniform'],
                          'tau_reio':     [0.01,  0.8,   'uniform'],
                          'n_s':          [0.9,   1.1,   'uniform'],
                          'A_s': [?,  ?,  'uniform'],
                           }"""
    log_prior_value = 0. #log_prior(parameters_and_priors, x)

    #Fix axion parameters
    x = np.insert(x, 6, [-26., 1.e-9])
    print('Parameters =', x)

    if log_prior_value == -np.inf:
        return log_prior_value, np.nan, np.nan, np.nan, -np.inf, -np.inf, log_prior_value
    else:
        log_likelihood_value, sigma8, delta_l_2, n_l = log_likelihood_Lyaf(x)
        #if log_likelihood_value == -np.inf:
        #    return -np.inf, sigma8, delta_l_2, n_l, -np.inf, log_likelihood_value, log_prior_value
        #else:
        log_likelihood_value_Planck = log_likelihood(x)
        if log_likelihood_value_Planck == -np.inf:
            return -1e+30, sigma8, delta_l_2, n_l, log_likelihood_value_Planck, log_likelihood_value, log_prior_value
        else:
            return log_prior_value + log_likelihood_value_Planck, np.array([sigma8, delta_l_2, n_l, log_likelihood_value_Planck, log_likelihood_value, log_prior_value]) #log_likelihood_value +

#Prepare for pyPolyChord
def log_likelihood_wrapper(theta):
    """Return the value of the log likelihood given x = theta."""
    return log_posterior(theta, parameters_and_priors)
'''mu = np.array([0.022242,  0.1,  0.673,  0.065,  0.9658,  3.0753, -25.5, 0.002, 1.0])
sigma_inv = 1. / (0.01 * mu)
#nDims = len(theta)
r2 = (theta - mu)**2
#logL = -np.log(2*np.pi*sigma*sigma)*nDims/2.0
logL = -sum(r2*sigma_inv*sigma_inv/2.)

return logL, [sum(r2)]
'''

def prior(x_hypercube):
    """Return the value of the prior given x defined on the unit (0, 1) hypercube."""
    x_transformed = np.zeros_like(x_hypercube)
    for i, key in enumerate(parameters_and_priors.keys()):
        if parameters_and_priors[key][2] == 'uniform':
            x_transformed[i] = ppp.UniformPrior(parameters_and_priors[key][0], parameters_and_priors[key][1])(x_hypercube[i])
        elif parameters_and_priors[key][2] == 'gaussian':
            x_transformed[i] = ppp.GaussianPrior(parameters_and_priors[key][0], parameters_and_priors[key][1])(x_hypercube[i])
        else:
            raise ValueError
    return x_transformed

#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

#| Initialise the settings
nDims = 7
nDerived = 6
settings = pps.PolyChordSettings(nDims, nDerived)
settings.file_root = 'test_Planck_LCDM2'
settings.nlive = 240
settings.do_clustering = False
settings.read_resume = False
settings.feedback = 1
settings.synchronous = False

'''ndim = 9
nwalkers = 120 

# set fiducial values and radius
FIDUCIAL = np.reshape(np.array([0.022242,  0.1,  0.673,  0.065,  0.9658,  3.0753, -25.5, 0.002, 1.0]).astype('float32'), (1, 9)) #-26.1, 0.006,
EPSILON  = np.reshape(np.array([1e-4,      1e-3,     1e-2,   1e-3,   1e-3,    1e-3, 1e-3, 1e-4, 1.e-3]).astype('float32'), (1,9)) #1E-3,      1E-4,




p0 = np.zeros((nwalkers, ndim))




for i in range(ndim):
    mean = FIDUCIAL[0][i]
    std = EPSILON[0][i]
    numbers = np.random.normal(loc = mean, scale = std, size = nwalkers)
    p0[:,i] = numbers
'''

#Set prior range for parameters
parameters_and_priors = {'omega_b':      [0.0174, 0.0274, 'uniform'],
                         'omega_cdm':    [1.e-32, 0.14,  'uniform'],
                         'h':            [0.55, 0.82,   'uniform'],
                         'tau_reio':     [0.065, 0.015,   'gaussian'],
                         'n_s':          [0.86, 1.07,   'uniform'],
                         'ln10^{10}A_s': [1.61,  3.4,  'uniform'],
                         'A_planck':     [1.0,   0.0025,  'gaussian']}
#                         'm_ax': [-27., -23., 'uniform'],
#                         'omega_ax': [1.e-32, 0.015,  'uniform'],

#Run pyPolyChord instead
output = pp.run_polychord(log_likelihood_wrapper, nDims, nDerived, settings, prior, dumper)

'''paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims+nDerived)]
#paramnames += [('r*', 'r')]
output.make_paramnames_files(paramnames)

import getdist.plots
posterior = output.posterior
g = getdist.plots.getSubplotPlotter()
g.triangle_plot(posterior, filled=True)
g.export('posterior_test_gaussian2.pdf')
'''

'''from multiprocessing import Pool
import pickle
with Pool(processes=60) as pool:
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "Planck_CMB_5000_axion_10pc.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[parameters_and_priors], pool=pool,backend=backend)
    start = time.time()
    max_n = 5000

    index = 0
    autocorr = np.empty(max_n)

    old_tau = np.inf
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        if not(sampler.iteration % 100):
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    samples_unflat = sampler.get_chain(flat=False)
    samples_flat = sampler.get_chain(flat=True)
    end = time.time()
    diff = end - start

    #Save sigma_8
    samples_sigma8_unflat = sampler.get_blobs()
    samples_sigma8_flat = sampler.get_blobs(flat=True)

    data_pkl = 'Planck_CMB_axionCAMB_5000_axion_10pc'+'.pkl'
    print('Dump data and time to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(samples_unflat,f)
    pickle.dump(samples_flat,f)
    pickle.dump(diff, f)

    #Save sigma_8
    pickle.dump(samples_sigma8_unflat,f)
    pickle.dump(samples_sigma8_flat,f)

    pickle.dump(sampler, f)
    pickle.dump(index, f)
    pickle.dump(autocorr, f)
    f.close() 
    import matplotlib.pyplot as plt

    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.savefig('convergence_test_Planck_CMB_5000_axion_10pc.png')
'''
