#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from test_log_likelihood import log_likelihood_Lyaf
from test_log_prior import log_prior
import emcee
import time
import os
from multiprocessing import Pool
import pickle
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
    log_prior_value = log_prior(parameters_and_priors, x)
    if log_prior_value == -np.inf:
        return -np.inf
    else:
        log_likelihood_value = log_likelihood_Lyaf(x)
        if log_likelihood_value == -np.inf:
            return -np.inf
        else:
            return log_prior_value + log_likelihood_value




ndim = 2
nwalkers = 120 

# set fiducial values and radius
FIDUCIAL = np.reshape(np.array([ -26., 0.006]).astype('float32'), (1, 2))
EPSILON  = np.reshape(np.array([ 1E-4,      1E-4]).astype('float32'), (1,2))




p0 = np.zeros((nwalkers, ndim))




for i in range(ndim):
    mean = FIDUCIAL[0][i]
    std = EPSILON[0][i]
    numbers = np.random.normal(loc = mean, scale = std, size = nwalkers)
    p0[:,i] = numbers

#Set prior range for parameters
parameters_and_priors = {#'omega_b':      [0.0174, 0.0274, 'uniform'],
                         #'omega_cdm':    [1.e-32, 0.14,  'uniform'],
                         #'h':            [0.55, 0.82,   'uniform'],
                         #'tau_reio':     [0.065, 0.015,   'gaussian'],
                         #'n_s':          [0.86, 1.07,   'uniform'],
                         #'ln10^{10}A_s': [1.61,  3.4,  'uniform'],
                         'm_ax': [-27.5, -22., 'uniform'],
                         'omega_ax': [1.e-32, 0.12011,  'uniform']}
#'yp':     [0.9,   1.1,  'uniform']}


from multiprocessing import Pool
import pickle
with Pool(processes=60) as pool:
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "Lyaf_eBOSS_500_wider_prior.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[parameters_and_priors], pool=pool,backend=backend)
    start = time.time()
    max_n = 500

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
    data_pkl = 'Lyaf_eBOSS_axionCAMB_500_wider_prior'+'.pkl'
    print('Dump data and time to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(samples_unflat,f)
    pickle.dump(samples_flat,f)
    pickle.dump(diff, f)
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
    plt.savefig('convergence_test_Lyaf_eBOSS_500_wider_prior.png')




