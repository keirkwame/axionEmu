The 3 files are for planck 2018 analysis sourced from axionCAMB and using EMCEE sampler. 
The test_log_prior file calculates the log(prior) given a set of input.
The test_log_likelihood file provides two methods to generate power spectra: If the input is omega_cdm, then set results = TTTEEE_generator(theta_[:7]).
If the input is omega_lamda, then set results = TTTEEE_generator_lamda(theta_[:7]). ('results' is a varible in log_likelihood function.) 
Remeber to change ini_file varible in either function to be the name of your params.ini file for axionCAMB.
Also, since we want to fix the value of axion mass, you need to input the value of axion massin 'lines[55] = 'm_ax = '+str(axion mass in eV)+'\n'' in either function.
To run planck 2018 analysis, please make sure to put all 3 files in the same directory as axionCAMB (i.e., the directory where you can run axionCAMB by './camb params.ini')
The prior range and fiducial values are set in test_log_posterior.py file. Run test_log_posterior.py to do planck 2018 analysis.
