# !git clone https://github.com/heatherprince/planck-lite-py.git

# move planck_lite_py.py out from /content/planck-lite-py to /content/planck_lite_py.py
from planck_lite_py import PlanckLitePy
import pyactlike
import numpy as np
import scipy.integrate as spi
import scipy.special as sps
import os

# get ls, Dltt, Dlte, Dlee
def TTTEEE_generator(theta):
    """
    Calculate ls, Dltt, Dlte, Dlee
    theta = ['omega_b','omega_cdm','H_0', 'tau_reio', 'n_s', 'ln10^{10}A_s', ('omega_ax'),'A_planck']
    return (ls, Dltt, Dlte, Dlee)
    """
    omega_b = float(theta[0])
    omega_cdm = float(theta[1])
    H_0 = float(theta[2])*100.0
    tau_reio = float(theta[3])
    n_s = float(theta[4])
    ln10A_s = float(theta[5])
    A_s = (np.exp(ln10A_s))/(10**10)
    omega_ax = float(theta[7])
    m_ax = 10. ** float(theta[6])
    ini_file = 'test_os_00' #CHANGE
    file = open(ini_file, 'r') #CHANGE
    lines = file.readlines()
    file.close()
    index = str(omega_b)+str(omega_cdm)+str(H_0)+str(A_s)+str(n_s)+str(tau_reio)+str(omega_ax)+str(m_ax)+str(np.random.rand(1)[0])
    lines[3] = 'output_root = test_'+index+'\n'
    lines[34] = 'ombh2 = '+str(omega_b)+'\n'
    lines[35] = 'omch2 = '+str(omega_cdm)+'\n'
    lines[38] = 'hubble         = '+str(H_0)+'\n'
    lines[91] = 'scalar_amp(1)             = '+str(A_s)+'\n'
    lines[92] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
    lines[106] = 're_optical_depth     = '+str(tau_reio)+'\n'
    lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
    lines[55] = 'm_ax = '+str(m_ax)+'\n' # input your axion mass value here! (replace number in str())
    new_file = 'test_os_likelihood' +index
    os.system('touch '+new_file)
    f1 = open(new_file, 'w')
    f1.writelines(lines)
    f1.close()
    predex = 'test_'+index+'_'
    try:
        os.system('./camb '+new_file)
        predex = 'test_'+index+'_'
        l_index, C_tt, C_ee, C_bb, C_te = np.loadtxt(predex+'lensedCls.dat', unpack = True)
        os.system('rm ' + predex + 'lensedCls.dat')
        #os.system('rm ' + predex + 'transfer_out.dat')
        os.system('rm ' + predex + 'scalCovCls.dat')
        os.system('rm ' + predex + 'scalCls.dat')
        #os.system('rm ' + predex + 'matterpower.dat')
        #os.system('rm ' + predex + 'lenspotentialCls.dat')
        os.system('rm ' + predex + 'params.ini')
        os.system('rm ' + new_file)
        return (l_index[:3001], C_tt[:3001], C_te[:3001], C_ee[:3001])
    except Exception:
        os.system('rm ' + new_file)
        os.system('rm ' + predex + 'params.ini')
        return -np.inf

# get ks, Pk_linear
def Pk_generator(theta):
    """
    Calculate ks, Pk
    theta = ['omega_b','omega_cdm','H_0', 'tau_reio', 'n_s', 'ln10^{10}A_s', 'log10(m_ax [eV])', ('omega_ax')]
    return (ks, Pk)
    """
    omega_b = float(theta[0])
    omega_cdm = float(theta[1])
    H_0 = float(theta[2])*100.0
    tau_reio = float(theta[3])
    n_s = float(theta[4])
    ln10A_s = float(theta[5])
    A_s = (np.exp(ln10A_s))/(10**10)
    m_ax = 10. ** float(theta[6])
    omega_ax = float(theta[7])
    ini_file = 'test_os_00_Pk_z3' #CHANGE
    file = open(ini_file, 'r') #CHANGE
    lines = file.readlines()
    file.close()
    index = str(omega_b)+str(omega_cdm)+str(H_0)+str(A_s)+str(n_s)+str(tau_reio)+str(m_ax)+str(omega_ax)+str(np.random.rand(1)[0])
    lines[3] = 'output_root = test_'+index+'\n'
    lines[34] = 'ombh2 = '+str(omega_b)+'\n'
    lines[35] = 'omch2 = '+str(omega_cdm)+'\n'
    lines[38] = 'hubble         = '+str(H_0)+'\n'
    lines[91] = 'scalar_amp(1)             = '+str(A_s)+'\n'
    lines[92] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
    lines[106] = 're_optical_depth     = '+str(tau_reio)+'\n'
    lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
    lines[55] = 'm_ax = '+str(m_ax)+'\n' # input your axion mass value here! (replace number in str())
    new_file = 'test_os_likelihood' +index
    os.system('touch '+new_file)
    f1 = open(new_file, 'w')
    f1.writelines(lines)
    f1.close()
    predex = 'test_'+index+'_'
    try:
        os.system('./camb '+new_file)
        predex = 'test_'+index+'_'
        k, Pk = np.loadtxt(predex+'matterpower_z3.dat', unpack = True)
        #os.system('rm ' + predex + 'lensedCls.dat')
        os.system('rm ' + predex + 'transfer_out_z3.dat')
        #os.system('rm ' + predex + 'scalCovCls.dat')
        #os.system('rm ' + predex + 'scalCls.dat')
        os.system('rm ' + predex + 'matterpower_z3.dat')
        #os.system('rm ' + predex + 'lenspotentialCls.dat')
        os.system('rm ' + predex + 'params.ini')
        os.system('rm ' + new_file)

        #Calculate sigma8
        k0, Pk0 = np.loadtxt(predex+'matterpower_z0.dat', unpack = True)
        sigma8 = get_sigma_r(k0, Pk0, r=8.)
        os.system('rm ' + predex + 'transfer_out_z0.dat')
        os.system('rm ' + predex + 'matterpower_z0.dat')

        return (k, Pk, sigma8)
    except Exception:
        os.system('rm ' + new_file)
        os.system('rm ' + predex + 'params.ini')
        return -np.inf

def get_sigma_r(k, Pk, r=8.):
    """Calculate sigma_r given matter power spectrum. r is in Mpc/h"""
    x = k * r
    #j1 = (np.sin(x) / x) - np.cos(x)
    j1 = sps.spherical_jn(1, x)
    Pk_dim = ((k ** 3.) * Pk) / (2. * (np.pi ** 2.))
    integrand = ((3. * j1 / x) ** 2.) * Pk_dim
    variance = spi.simpson(integrand, x=np.log(k))
    sigma_r = np.sqrt(variance)
    print('Sigma_r =', sigma_r)
    return sigma_r

def TTTEEE_generator_lamda(theta):
    """
    Calculate ls, Dltt, Dlte, Dlee
    theta = ['omega_b','omega_lamda','H_0', 'tau_reio', 'n_s', 'ln10^{10}A_s', ('omega_ax'),'A_planck']
    return (ls, Dltt, Dlte, Dlee)
    """
    omega_b = float(theta[0])
    omega_lamda = float(theta[1])
    H_0 = float(theta[2])*100.0
    tau_reio = float(theta[3])
    n_s = float(theta[4])
    ln10A_s = float(theta[5])
    A_s = (np.exp(ln10A_s))/(10**10)
    omega_ax = float(theta[6])
    omega_cdm = float(theta[2])**2 - (omega_lamda+omega_ax+omega_b+0.0006)
    ini_file = 'test_os_00_5e5' # input your ini file
    file = open(ini_file, 'r') #CHANGE
    lines = file.readlines()
    file.close()
    index = str(omega_b)+str(omega_cdm)+str(H_0)+str(A_s)+str(n_s)+str(tau_reio)+str(omega_ax)+str(np.random.rand(1)[0])
    lines[3] = 'output_root = test_'+index+'\n'
    lines[34] = 'ombh2 = '+str(omega_b)+'\n'
    lines[35] = 'omch2 = '+str(omega_cdm)+'\n'
    lines[38] = 'hubble         = '+str(H_0)+'\n'
    lines[91] = 'scalar_amp(1)             = '+str(A_s)+'\n'
    lines[92] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
    lines[106] = 're_optical_depth     = '+str(tau_reio)+'\n'
    lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
    lines[55] = 'm_ax = '+str(1e-30)+'\n' # input your axion mass value here! (replace number in str())
    new_file = 'test_os_likelihood' +index
    os.system('touch '+new_file)
    f1 = open(new_file, 'w')
    f1.writelines(lines)
    f1.close()
    predex = 'test_'+index+'_'
    try:
        os.system('./camb '+new_file)
        predex = 'test_'+index+'_'
        l_index, C_tt, C_ee, C_bb, C_te = np.loadtxt(predex+'lensedCls.dat', unpack = True)
        os.system('rm ' + predex + 'lensedCls.dat')
        #os.system('rm ' + predex + 'transfer_out.dat')
        os.system('rm ' + predex + 'scalCovCls.dat')
        os.system('rm ' + predex + 'scalCls.dat')
        #os.system('rm ' + predex + 'matterpower.dat')
        os.system('rm ' + predex + 'lenspotentialCls.dat')
        os.system('rm ' + predex + 'params.ini')
        os.system('rm ' + new_file)
        return (l_index[:2507], C_tt[:2507], C_te[:2507], C_ee[:2507])
    except Exception:
        os.system('rm ' + new_file)
        os.system('rm ' + predex + 'params.ini')
        return -np.inf

def log_likelihood(theta_):
    """
    Calculate the log_likelihood axionCAMB and PlanckLitePy package. If the input is omega_cdm, then use TTTEEE_generator. If the input is omega_lamda, then use TTTEEE_generator_lamda.
    """
    A_plank = theta_[-1]
    results = TTTEEE_generator(theta_[:-1])
    if results == -np.inf:
        return -np.inf
    else:
        l_index, C_tt, C_te, C_ee = results
        C_tt = C_tt/(A_plank**2)
        C_te = C_te/(A_plank**2)
        C_ee = C_ee/(A_plank**2)
        TTTEEE2018_lowTTbins=PlanckLitePy(data_directory='/home/keir/Software/planck-lite-py/data', year=2018, spectra='TTTEEE', use_low_ell_bins=False) #CHANGE
        loglike=TTTEEE2018_lowTTbins.loglike(C_tt, C_te, C_ee, int(l_index[0]))
        return loglike

def log_likelihood_ACT_DR4(theta_):
    """
    Calculate the log_likelihood axionCAMB and pyactlike package. If the input is omega_cdm, then use TTTEEE_generator. If the input is omega_lamda, then use TTTEEE_generator_lamda.
    """
    yp = theta_[7]
    results = TTTEEE_generator(theta_[:7])
    if results == -np.inf:
        return -np.inf
    else:
        l_index, C_tt, C_te, C_ee = results
        #C_tt = C_tt/(A_plank**2)
        #C_te = C_te/(A_plank**2)
        #C_ee = C_ee/(A_plank**2)
        like = pyactlike.ACTPowerSpectrumData()
        loglike=like.loglike(C_tt, C_te, C_ee, yp)
        print('Log likelihood =', loglike)
        #TTTEEE2018_lowTTbins.loglike(C_tt, C_te, C_ee, int(l_index[0]))
        return loglike

def log_likelihood_Lyaf(theta_):
    """
    Calculate the log_likelihood axionCAMB and compressed Lyaf eBOSS likelihood.
    """
    #yp = theta_[7]

    #Add LCDM parameters
    #theta_extended = np.concatenate((np.array([0.022383, 0.12011 - theta_[1], 0.6732, 0.0543, 0.96605, 3.0448]), theta_))
    theta_extended = theta_[:-1]

    results = Pk_generator(theta_extended)
    if results == -np.inf:
        return -np.inf, np.nan, np.nan, np.nan
    else:
        k, Pk, sigma8 = results
        #C_tt = C_tt/(A_plank**2)
        #C_te = C_te/(A_plank**2)
        #C_ee = C_ee/(A_plank**2)
        #like = pyactlike.ACTPowerSpectrumData()
        #loglike=like.loglike(C_tt, C_te, C_ee, yp)

        index_pivot = 461 #CHANGE!!
        print(k[index_pivot])
        delta_l_2 = Pk[index_pivot] * (k[index_pivot] ** 3.) / (2. * (np.pi ** 2.))
        n_l = (np.log(Pk[index_pivot]) - np.log(Pk[index_pivot - 1])) / (np.log(k[index_pivot]) - np.log(k[index_pivot - 1]))

        #Data
        delta_l_2_eBOSS = 0.310
        delta_l_2_eBOSS_sigma = 0.020
        n_l_eBOSS = -2.340
        n_l_eBOSS_sigma = 0.006
        rho = 0.512

        delta_x = (delta_l_2 - delta_l_2_eBOSS) / delta_l_2_eBOSS_sigma
        delta_y = (n_l - n_l_eBOSS) / n_l_eBOSS_sigma
        loglike = (-1. * ((delta_x ** 2.) - (2. * rho * delta_x * delta_y) + (delta_y ** 2.))) / (2. * (1. - (rho ** 2.)))

        print('Log likelihood =', loglike)
        #TTTEEE2018_lowTTbins.loglike(C_tt, C_te, C_ee, int(l_index[0]))
        return loglike, sigma8, delta_l_2, n_l
