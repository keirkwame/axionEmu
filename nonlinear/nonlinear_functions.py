# Composite code
# Run axionHMCode and axionCAMB
# Calculate the lensing from linear
# and non-linear power spectra
# Note: Need modified axionCAMB which outputs the Weyl potential
# Also need the (regular) camb package for cosmological calculations

# Regular imports
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

# Camb import
import camb

# Import axionHMCode
import sys
import os

sys.path.append('../axionHMcode/axionCAMB_and_lin_PS/')
sys.path.append('../axionHMcode/cosmology/')
sys.path.append('../axionHMcode/axion_functions/')
sys.path.append('../axionHMcode/halo_model/')
sys.path.append('../axionHMcode/')
from axionCAMB_and_lin_PS import axionCAMB_wrapper 
from axionCAMB_and_lin_PS import load_cosmology  
from axionCAMB_and_lin_PS import lin_power_spectrum 
from axionCAMB_and_lin_PS import PS_interpolate 

from halo_model import HMcode_params
from halo_model import PS_nonlin_cold
from halo_model import PS_nonlin_axion

from axion_functions import axion_params


def get_limber_clkk_flat_universe(results, interp_pk, lmax, kmax, nz, zsrc=None):
    # Adapting code from Antony Lewis' CAMB notebook
    if zsrc is None:
        chistar = results.conformal_time(0)- results.tau_maxvis
    else:
        chistar = results.comoving_radial_distance(zsrc)
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    
    #Get lensing window function (flat universe)
    win = ((chistar-chis)/(chis**2*chistar))**2
    #Do integral over chi
    ls = np.arange(0,lmax+2, dtype=np.float64)
    cl_kappa=np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero 
                            # k values out of range of interpolation
    for i, l in enumerate(ls[2:]):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        cl_kappa[i+2] = np.dot(dchis, w*interp_pk(zs, k, grid=False)*win/k**4)
    cl_kappa*= (ls*(ls+1))**2
    return cl_kappa


def do_non_linear_lensing(H0, omch2, ombh2, As, ns, m_ax, omaxh2, zs, T_path, unlensed_cls):
    '''
    Compute the non-linear matter power spectra, lensed cls, and lensing potential using axionHMCode
    
    Inputs:
    first five inputs are the regular cosmological parameters
    m_ax, omaxh2 are the axion parameters (mass in eV and relic density)
    zs are the redshifts (must be increasing)
    T_path is the directory where the previously generated transfer functions are (same redshifts as zs)
    unlensed cls are the unlensed TT, EE, TE
    
    Outputs:
    lensed cls using non-linear spectra
    lensing potential cls
    '''
    
    # Initialize parameters
    cosmos = {}
    
    cosmos['As'] = As
    cosmos['ns'] = ns
    cosmos['h'] = H0 / 100
    cosmos['omega_b_0'] = ombh2
    cosmos['omega_d_0'] = omch2 # unconventional notation!!
    cosmos['omega_ax_0'] = omaxh2
    cosmos['m_ax'] = m_ax
    cosmos['omega_db_0'] = cosmos['omega_b_0'] + cosmos['omega_d_0']
    cosmos['omega_m_0'] = cosmos['omega_db_0'] + cosmos['omega_ax_0']

    cosmos['Omega_b_0']     = cosmos['omega_b_0'] / cosmos['h']**2
    cosmos['Omega_d_0']     = cosmos['omega_d_0'] / cosmos['h']**2
    cosmos['Omega_db_0']    = cosmos['omega_db_0'] / cosmos['h']**2
    cosmos['Omega_m_0']     = cosmos['omega_m_0'] / cosmos['h']**2
    cosmos['Omega_w_0']     = 1 - cosmos['Omega_m_0']

    cosmos['M_min'] = 7
    cosmos['M_max'] = 18
    cosmos['k_piv'] = 0.05
    cosmos['z']     = 0.
    cosmos['transfer_kmax'] = 20

    # Compute camb background cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omdh2, mnu=0.06, omk=0)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    r = camb.get_results(pars)

    # Read in Tk and Weyl potential
    weyl_trans_dic_LCDM = []
    kh_trans_dic_LCDM   = []
    for i in range(1, len(cosmos_LCDM['z'])+1)[::-1]:
        Tk = np.loadtxt(T_path+str(i)+'.dat')
        kh_trans_dic_LCDM.append(Tk[:,0])
        weyl_trans_dic_LCDM.append(Tk[:,-2])
    
    kh_trans_dic_LCDM = np.array(kh_trans_dic_LCDM)
    weyl_trans_dic_LCDM = np.array(weyl_trans_dic_LCDM)

    # Weyl Pk from primordial Pk
    h = cosmos['h']
    kh = transfer_table[:,0]
    k = kh * h
    Pk0 = As * (k/0.05)**(ns-1)*kh**4 / (kh**3/(2*np.pi**2)) *h
    weyl_Pk_lin = Pk0 * weyl_trans_list**2
    
    # Compute non-linear using axionHMCode
    power_spec_dic = []
    PkNL_list = []
    M_arr = np.logspace(8, 17, 100)

    for i in range(len(zs)):
        cosmos_specific_z = cosmos.copy()
        cosmos_specific_z['z'] = zs[i]
    
        power_spec_dic = lin_power_spectrum.func_power_spec_dic(T_path + str(i+3)+'.dat',
                                                                          cosmos_specific_z) # i+3 since first two already taken 
        
        power_spec_interp_dic = lin_power_spectrum.func_power_spec_interp_dic(power_spec_dic[i], 
                                                                                        cosmos_specific_z)
        hmcode_params = HMcode_params.HMCode_param_dic(cosmos, 
                                                       power_spec_interp_dic['k'], 
                                                       power_spec_interp_dic['cold'])
        
        PS_matter_nonlin = PS_nonlin_axion.full_halo_model_ax(M_arr, 
                                                              power_spec_dic[i]['k'], 
                                                              power_spec_dic[i]['power_total'], 
                                                              power_spec_interp_dic['k'], 
                                                              power_spec_interp_dic['cold'], 
                                                              cosmos_specific_z, 
                                                              hmcode_params, 
                                                              cosmos['Omega_m_0'], 
                                                              cosmos['Omega_db_0'], 
                                                              alpha = True, 
                                                              eta_given = False, 
                                                              one_halo_damping = True, 
                                                              two_halo_damping = True)
        PkNL_list.append(PS_matter_nonlin[0])
    
    # Non-linear transfer function
    PkL_list = np.array([power_spec_dic[::-1][i]['power_total'] for i in range(len(zs))]) # change order for increasing z
    PkNL_list = np.array(PkNL_list[::-1])

    TkNL2 = PkNL_list/PkL_list
        
    # Create interpolator for Limber integral
    P_weyl_NL = RectBivariateSpline(zs, k, TkNL2 * weyl_Pk_lin)
    
    # Perform Limber integral
    clkk_NL = get_limber_clkk_flat_universe(r, P_weyl_NL, lmax, cosmos_LCDM['transfer_kmax'], 100, zsrc=None)
    
    # Change to phi
    clpp_NL = 4*clkk_NL/2/np.pi
    
    # Recompute lensed cls "manually"
    lensed_cls_corr = camb.correlations.lensed_cls(cls, clpp_NL) # without theta_max=None takes much shorter but diverges at l>3500
    
    clTT, clEE, clTE = lensed_cls_corr[:,0], lensed_cls_corr[:,1], lensed_cls_corr[:,3]
    
    return clTT, clEE, clTE, clpp_NL


def do_nonlinear_pk(H0, omch2, ombh2, As, ns, m_ax, omaxh2, z, T_path):
    '''
    Compute non-linear power spectrum from linear transfer functions
    same parameters as do_nonlinear_lensing except for Cls
    return single Pk
    '''
        
    #Initialize parameters
    cosmos = {}

    cosmos['As'] = As
    cosmos['ns'] = ns
    cosmos['h'] = H0 / 100
    cosmos['omega_b_0'] = ombh2
    cosmos['omega_d_0'] = omch2 # unconventional notation!!
    cosmos['omega_ax_0'] = omaxh2
    cosmos['m_ax'] = m_ax
    cosmos['omega_db_0'] = cosmos['omega_b_0'] + cosmos['omega_d_0']
    cosmos['omega_m_0'] = cosmos['omega_db_0'] + cosmos['omega_ax_0']

    cosmos['Omega_b_0']     = cosmos['omega_b_0'] / cosmos['h']**2
    cosmos['Omega_d_0']     = cosmos['omega_d_0'] / cosmos['h']**2
    cosmos['Omega_db_0']    = cosmos['omega_db_0'] / cosmos['h']**2
    cosmos['Omega_m_0']     = cosmos['omega_m_0'] / cosmos['h']**2
    cosmos['Omega_w_0']     = 1 - cosmos['Omega_m_0']

    cosmos['M_min'] = 7
    cosmos['M_max'] = 18
    cosmos['k_piv'] = 0.05
    cosmos['z']     = z
    cosmos['transfer_kmax'] = 20


    power_spec_dic = lin_power_spectrum.func_power_spec_dic(T_path + '.dat', cosmos)
    power_spec_interp_dic = lin_power_spectrum.func_power_spec_interp_dic(power_spec_dic, cosmos)
    hmcode_params = HMcode_params.HMCode_param_dic(cosmos,
                                                   power_spec_interp_dic['k'],
                                                   power_spec_interp_dic['cold'])

    PS_matter_nonlin = PS_nonlin_axion.full_halo_model_ax(M_arr,
                                                          power_spec_dic['k'],
                                                          power_spec_dic['power_total'],
                                                          power_spec_interp_dic['k'],
                                                          power_spec_interp_dic['cold'],
                                                          cosmos_specific_z,
                                                          hmcode_params,
                                                          cosmos['Omega_m_0'],
                                                          cosmos['Omega_db_0'],
                                                          alpha = True,
                                                          eta_given = False,
                                                          one_halo_damping = True,
                                                          two_halo_damping = True)
    return PS_matter_nonlin[0]
