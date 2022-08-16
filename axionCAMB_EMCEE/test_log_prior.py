#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
# Hide GPU from visible devices 
tf.config.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp

def log_prior(parameters_and_priors, theta):
    """
    Calculate log(prior). Codes adapted from cosmopower (tf_planck2018_lite.py)
    """
    model = []
    for elem in parameters_and_priors:
        low,high,name = parameters_and_priors[elem]
        if name == 'uniform':
            model.append(tfp.distributions.Uniform(low=low, high=high))
        elif name=='gaussian':
            model.append(tfp.distributions.Normal(loc=low, scale=high))
    priors_value = tfp.distributions.Blockwise(model)
    pr = priors_value.prob(theta)
    logpr = tf.math.log(pr)
    return logpr.numpy()

