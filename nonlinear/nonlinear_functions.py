# Composite code
# Run axionHMCode and axionCAMB
# Calculate the lensing from linear
# and non-linear power spectra
# Note: Need modified axionCAMB which outputs the Weyl potential
# ^ no longer true: getting Weyl spectrum from pyCAMB
# Also need the (regular) pyCAMB package for cosmological calculations

# Regular imports
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

# Camb import
import camb
from camb import correlations, model

# Import axionHMCode
import sys
import os
hmcode_path = '/home/keir/Software/axionHMcode_fast/'
sys.path.append(hmcode_path + 'axionCAMB_and_lin_PS/')
sys.path.append(hmcode_path + 'cosmology/')
sys.path.append(hmcode_path + 'axion_functions/')
sys.path.append(hmcode_path + 'halo_model/')
sys.path.append(hmcode_path)
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


def do_non_linear_lensing(H0, omch2, ombh2, As, ns, m_ax, omaxh2, gamma_1, gamma_2, zs, T_path, return_matter_power=False):
    '''
    Compute the non-linear matter power spectra, lensed cls, and lensing potential using axionHMCode
    
    Inputs:
    first five inputs are the regular cosmological parameters
    m_ax, omaxh2 are the axion parameters (mass in eV and relic density)
    gamma_1, gamma_2 are the concentration-mass relation nuisance parameters
    zs are the redshifts (must be increasing)
    T_path is the directory where the previously generated transfer functions are (same redshifts as zs)
    
    Outputs:
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
    cosmos['Omega_ax_0']    = cosmos['omega_ax_0'] / cosmos['h']**2
    cosmos['Omega_m_0']     = cosmos['omega_m_0'] / cosmos['h']**2
    cosmos['Omega_w_0']     = 1 - cosmos['Omega_m_0']

    cosmos['M_min'] = 7
    cosmos['M_max'] = 18
    cosmos['k_piv'] = 0.05
    cosmos['z']     = 0.
    cosmos['transfer_kmax'] = 20

    cosmos['gamma_1'] = gamma_1
    cosmos['gamma_2'] = gamma_2

    if not return_matter_power:
        # Compute camb background cosmology
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=(omch2+omaxh2), mnu=0.06, omk=0) # account for axion DM
        pars.set_for_lmax(3000, lens_potential_accuracy=4, nonlinear=False, )
        pars.InitPower.set_params(As=As, ns=ns, r=0)
        #r = camb.get_background(pars)
        r = camb.get_results(pars) # also run Pk calculation

        # Weyl power spectrum for CDM
        Pweyl_cdm = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                   hubble_units=True, k_hunit=True, kmax=cosmos['transfer_kmax'],
                                                   var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])

        # Linear matter power spectrum for CDM
        mpk_cdm = camb.get_matter_power_interpolator(pars, nonlinear=False,
                                                 hubble_units=True, k_hunit=True, kmax=cosmos['transfer_kmax'],
                                                 var1=model.Transfer_tot,var2=model.Transfer_tot, zmax=zs[-1])
        
        # Now computed later due to camb version clash
        '''
        # Read in Tk and Weyl potential
        weyl_trans_list = []
        kh_trans_list   = []
        for i in range(1, len(zs)+1)[::-1]:
            print('Loading Weyl transfer file:', T_path+str(i)+'.dat')
            Tk = np.loadtxt(T_path+str(i)+'.dat')
            kh_trans_list.append(Tk[:,0])
            weyl_trans_list.append(Tk[:,-2])
        
        kh_trans_list = np.array(kh_trans_list)
        weyl_trans_list = np.array(weyl_trans_list)

        # Weyl Pk from primordial Pk
        h = cosmos['h']
        kh = kh_trans_list[0]
        k = kh * h
        Pk0 = As * (k/0.05)**(ns-1)*kh**4 / (kh**3/(2*np.pi**2)) *h
        weyl_Pk_lin = Pk0 * weyl_trans_list**2
        '''

    # Compute non-linear using axionHMCode
    PkL_list = []
    PkNL_list = []
    PkL_list_cdm   = []
    Pweyl_list_cdm = []
    k_list = []
    M_arr = np.logspace(8, 17, 100)
    
    for i in range(len(zs)):
        cosmos_specific_z = cosmos.copy()
        cosmos_specific_z['z'] = zs[::-1][i] #Looping through decreasing redshift

        if not return_matter_power:
            transfer_file_name = T_path + str(i+1)+'.dat'
        else:
            transfer_file_name = T_path + '.dat'
        print('Transfer file name', transfer_file_name, cosmos_specific_z['z'])

        power_spec_dic = lin_power_spectrum.func_power_spec_dic(transfer_file_name,
                                                                          cosmos_specific_z)
        
        # Add axion lin pk, cdm lin pk, and Weyl pk for cdm
        PkL_list.append(power_spec_dic['power_total'])
        PkL_list_cdm.append(mpk_cdm.P(cosmos_specific_z['z'], power_spec_dic['k']))
        Pweyl_list_cdm.append(Pweyl_cdm.P(cosmos_specific_z['z'], power_spec_dic['k']))
        #k_list.append(power_spec_dic['k'])
        
        if cosmos_specific_z['z'] >= 6.5: #zs[i] >= 6.5:
            PkNL_list.append(power_spec_dic['power_total'])
        
        else:
            power_spec_interp_dic = lin_power_spectrum.func_power_spec_interp_dic(power_spec_dic, 
                                                                                        cosmos_specific_z)
            hmcode_params = HMcode_params.HMCode_param_dic(cosmos_specific_z, 
                                                           power_spec_interp_dic['k'], 
                                                           power_spec_interp_dic['cold'])
        
            axion_param = axion_params.func_axion_param_dic(M_arr, cosmos_specific_z, power_spec_interp_dic, eta_given=False)
            PS_matter_nonlin = PS_nonlin_axion.func_full_halo_model_ax(M_arr, 
                                                                       power_spec_dic, 
                                                                       power_spec_interp_dic, 
                                                                       cosmos_specific_z, 
                                                                       hmcode_params, 
                                                                       axion_param, 
                                                                       alpha = True, 
                                                                       eta_given = False, 
                                                                       one_halo_damping = True, 
                                                                       two_halo_damping = True)
            PkNL_list.append(PS_matter_nonlin[0])
    
    # Non-linear transfer function
    PkL_list = np.array(PkL_list[::-1]) #Increasing redshift
    PkNL_list = np.array(PkNL_list[::-1])
    PkL_list_cdm   = np.array(PkL_list_cdm[::-1])
    Pweyl_list_cdm = np.array(Pweyl_list_cdm[::-1])

    debug = False
    if debug:
        np.savetxt('k_list.dat', power_spec_dic['k'])
        np.savetxt('PkL_list.dat', PkL_list)
        np.savetxt('PkNL_list.dat', PkNL_list)
        np.savetxt('PkL_list_cdm.dat', PkL_list_cdm)
        np.savetxt('Pweyl_list_cdm.dat', Pweyl_list_cdm)


    Tkax2 = PkL_list/PkL_list_cdm # captures the effects of axions
    TkNL2 = PkNL_list/PkL_list # catpures the effects of non-linearities 

    # Compute Weyl now as ratio to cdm
    weyl_Pk_lin = Tkax2 * Pweyl_list_cdm
    
    # Checking for normalization errors
    tol   = 1e-3
    k_ind = (power_spec_dic['k'] < 1e-3) & (power_spec_dic['k'] > 1e-4) # large-scales for testing
    err   = Tkax2.T[k_ind] - 1.
    assert np.max(np.abs(err)) < tol

    if not return_matter_power:
        # Create interpolator for Limber integral
        P_weyl_NL = RectBivariateSpline(zs, power_spec_dic['k']*cosmos['h'], TkNL2 * weyl_Pk_lin/cosmos['h']**3) # changing to non-Hubble units
        
        # Perform Limber integral
        clkk_NL = get_limber_clkk_flat_universe(r, P_weyl_NL, 6300, cosmos['transfer_kmax'], 100, zsrc=None)
        
        # Change to phi
        clpp_NL = 4*clkk_NL/2/np.pi
        
        return clpp_NL, PkNL_list, PkL_list, power_spec_dic['k'], zs, weyl_Pk_lin
    else:
        print(len(PkNL_list[0]))
        return PkNL_list[0], power_spec_dic['k'], power_spec_interp_dic['k']


def do_nonlinear_pk(H0, omch2, ombh2, As, ns, m_ax, omaxh2, gamma_1, gamma_2, z, T_path):
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

    cosmos['gamma_1'] = gamma_1
    cosmos['gamma_2'] = gamma_2

    M_arr = np.logspace(8, 17, 100)

    print('Generated dictionary')
    print('Loading this transfer file (for redshift):', T_path + '.dat', cosmos['z'])
    power_spec_dic = lin_power_spectrum.func_power_spec_dic(T_path + '.dat', cosmos)
    print('Linear power')
    power_spec_interp_dic = lin_power_spectrum.func_power_spec_interp_dic(power_spec_dic, cosmos)
    print('Linear power interpolated')
    hmcode_params = HMcode_params.HMCode_param_dic(cosmos,
                                                   power_spec_interp_dic['k'],
                                                   power_spec_interp_dic['cold'])
    print('HMcode parameters')
    #print(M_arr, cosmos, power_spec_interp_dic)
    axion_param = axion_params.func_axion_param_dic(M_arr, cosmos, power_spec_interp_dic, eta_given=False)
    print('Preliminary calculations')
    PS_matter_nonlin = PS_nonlin_axion.func_full_halo_model_ax(M_arr, power_spec_dic, power_spec_interp_dic, cosmos, hmcode_params, axion_param, alpha = True, eta_given = False, one_halo_damping = True, two_halo_damping = True)
    print('Calculated halo model power')

    return PS_matter_nonlin[0]
