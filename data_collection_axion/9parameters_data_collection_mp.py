# In[ ]:


import os
import numpy as np
import pickle
import copy
from time import time
from multiprocessing import Pool

import sys
sys.path.append('../nonlinear')
from nonlinear_functions import *

from camb import correlations

def data_collection(input):
    """
    LHD_para_pklfile = 'LHD_parameters_10000_0.pkl',
    index_pkl_name = '10000_0'
    params_ini_file = 'test_os_0'
    pre_index = 'test_' ("test_lensedCls.dat") 
    """
    (LHD_para_pklfile, index_pkl_name, params_ini_file, pre_index) = input
    f = open(LHD_para_pklfile,'rb')
    params = pickle.load(f)
    f.close()
    start = time()
    collection = {}
    collection_1 = {}
    collection_2 = {}
    collection_3 = {}
    collection_4 = {}
    collection['l_index'] = []
    collection['C_tt'] = []
    collection['C_ee'] = []
    collection['C_bb'] = []
    collection['C_te'] = []
    collection['C_phi'] = []
    collection_1['k_index'] = []
    collection_1['matter_mg'] = []
    collection_2['k_index'] = []
    collection_2['matter_mg'] = []

    #Test arrays
    collection['PkNL'] = []
    collection['PkL'] = []
    collection['k_out'] = []
    collection['k_out2'] = []
    collection['z_out'] = []
    collection['weyl'] = []

    # AL modif (make space for NL Pk)
    collection_3['k_index'] = []
    collection_3['k_indexb'] = []
    collection_3['matter_mg'] = []
    collection_4['k_index'] = []
    collection_4['k_indexb'] = []
    collection_4['matter_mg'] = []

    t_params = dict()
    t_params1 = dict()
    t_params2 = dict()
    t_params3 = dict()
    t_params4 = dict()
    
    # AL add z_lens generated from comoving distance
    z_lens = np.loadtxt('../nonlinear/optimal_z_array.dat')

    for key in params:
        t_params[key] = []
        t_params1[key]=[]
        t_params2[key] = []
        t_params3[key]=[]
        t_params4[key] = []
#     var_param = ["ombh2", "omch2", "re_optical_depth    ", "hubble        ", "scalar_spectral_index(1) ", "scalar_amp(1)            "]
    params_keys = ['omega_b', 'omega_cdm', 'tau_reio', 'H_0','n_s', 'A_s', 'ma', 'omega_ax', 'z', 'gamma_1', 'gamma_2']
    s = 0
    r = 0
    problem_list = dict()
    problem_list_r = dict()
    for key in params:
        problem_list[key] = []
        problem_list_r[key] = []
    for i in range(len(params['omega_b'])):
        omega_b = float(params['omega_b'][i])
        omega_cdm = float(params['omega_cdm'][i])
        H_0 = float(params['H_0'][i])
        tau_reio = float(params['tau_reio'][i])
        n_s = float(params['n_s'][i])
        A_s = float(params['A_s'][i])
        ma = float(params['ma'][i])
        omega_ax = float(params['omega_ax'][i])
        z1 = float(params['z'][i])
        z2 = 5.0-float(z1)

        if z1 in z_lens:
            z1 += 1.e-5
        if z2 in z_lens:
            z2 += 1.e-5

        gamma_1 = float(params['gamma_1'][i])
        gamma_2 = float(params['gamma_2'][i])
        print('gamma_1, gamma_2 =', gamma_1, gamma_2)
        pre_name = pre_index + str(i) 
        try:
            print("############# " + params_ini_file)
            f1 = open(params_ini_file, 'r') #CHANGE
            lines = f1.readlines()
            f1.close()
            print('Opened and closed params.ini')
            lines[3] = 'output_root = ' + pre_name + '\n'
            lines[34] = 'ombh2 = '+str(omega_b)+'\n'
            lines[35] = 'omch2 = '+str(omega_cdm)+'\n'
            lines[38] = 'hubble         = '+str(H_0)+'\n'
            lines[91] = 'scalar_amp(1)             = '+str(A_s)+'\n'
            lines[92] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
            lines[106] = 're_optical_depth     = '+str(tau_reio)+'\n'
            lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
            lines[55] = 'm_ax = '+str(ma)+'\n'

            ##Find redshift index for z1 and z2
            '''z_combined = np.concatenate((z_lens[::-1], np.array([z1, z2])))
            print(z_combined)
            z_combined_indices = np.argsort(z_combined)[::-1] + 1 #CAMB lists starting at index = 1 #Index for decreasing z
            print(z_combined_indices)
            print(z_combined[z_combined_indices - 1])
            z1_idx = z_combined_indices[-2]
            z2_idx = z_combined_indices[-1]
            zlens_idx = z_combined_indices[:-2]
            print(zlens_idx, z1_idx, z2_idx)
            print('Printed indices!')
            print(np.concatenate((z_lens[::-1], np.array([z1, z2]))), np.concatenate((z_lens[::-1], np.array([z1, z2])))[z_combined_indices])'''
            assert z2 < z1
            z1_idx = np.searchsorted(-1. * z_lens[::-1], -1. * z1) #+ 1
            z2_idx = np.searchsorted(-1. * z_lens[::-1], -1. * z2) + 1 #Add 1 to account for z1 being inserted
            #zlens_idx = np.delete(np.arange(z_lens.shape[0] + 2), [z1_idx - 1, z2_idx - 1]) + 1
            print('New indices for Latin hypercube redshifts', z1_idx, z2_idx)

            z_combined = np.concatenate((z_lens[::-1], np.array([z1, z2])))
            z_combined_indices = np.argsort(z_combined)[::-1]
            z_ordered = z_combined[z_combined_indices]
            print('Re-ordered and combined redshifts', z_combined_indices, z_ordered)

            lines[148] = 'transfer_num_redshifts  = '+str(len(z_lens) + 2)+'\n'
            #lines[150] = 'transfer_redshift('+ str(z1_idx) +')    = '+str(z1)+'\n'
            #lines[152] = 'transfer_redshift('+ str(z2_idx) +')    = '+str(z2)+'\n'
            lines[150] = '\n'
            lines[152] = '\n'
            '''lines[151] = 'transfer_filename('+ str(z1_idx) +')    = transfer_'+str(len(z_lens) + 1)+'.dat\n'
            lines[153] = 'transfer_filename('+ str(z2_idx) +')    = transfer_'+str(len(z_lens) + 2)+'.dat\n'
            lines[155] = 'transfer_matterpower('+ str(z1_idx) +')    = matterpower_'+str(len(z_lens) + 1)+'.dat\n'
            lines[156] = 'transfer_matterpower('+ str(z2_idx) +')    = matterpower_'+str(len(z_lens) + 2)+'.dat\n'
            '''
            lines[151] = '\n'
            lines[153] = '\n'
            lines[155] = '\n'
            lines[156] = '\n'
            
            #if 'transfer_redshift(3)    = '+str(z_lens[::-1][2])+'\n' in lines: # make sure it's not already there
            #    print('Already done transfer list')
            #else:
            for j in range(len(z_lens) + 2): #AL modif add Nz transfer calc
                print('Saving transfer_'+str(z_combined_indices[j] + 1)+'.dat;', z_ordered[j])
                lines.extend('transfer_redshift(' + str(j+1) + ')    = '+str(z_ordered[j])+'\n')
                lines.extend('transfer_filename(' + str(j+1) + ')    = transfer_'+str(z_combined_indices[j] + 1)+'.dat\n')
                #if (j == z1_idx) or (j == z2_idx):
                lines.extend('transfer_matterpower(' + str(j+1) + ')    = matterpower_'+str(z_combined_indices[j] + 1)+'.dat\n')

            #os.system('rm -r '+params_ini_file)
            #os.system('touch '+params_ini_file)
            os.system('cp ' + params_ini_file + ' ' + params_ini_file + '_copy')
            f2 = open(params_ini_file + '_copy', 'w')
            f2.writelines(lines)
            f2.close()

            ## RUN CAMB ##
            os.system('./camb_ifort '+params_ini_file + '_copy')
            print('Finished running CAMB')
            
            ## COLLECT DATA ##
            unlensed_cls =  np.loadtxt(pre_name+'_'+'scalCls.dat')
            l_index = unlensed_cls[:,0]
            k_index, matter_mg = np.loadtxt(pre_name+'_'+'matterpower_' + str(len(z_lens) + 1) + '.dat', unpack = True)
            k_index2, matter_mg2 = np.loadtxt(pre_name+'_'+'matterpower_' + str(len(z_lens) + 2) +'.dat', unpack = True)

            os.system('rm ' + params_ini_file + '_copy')
            os.system('rm -r '+pre_name+'_'+'params.ini')
            
            ## PERFORM NON-LINEAR TRANSFORMS & LENSING ##
            T_path = pre_name+'_'+'transfer_'
            ucls = np.zeros((len(l_index)+2, 4))
            C_tt = unlensed_cls[:,1]
            C_ee = unlensed_cls[:,2]
            C_te = unlensed_cls[:,3]
            
            ucls[:,0][2:] = C_tt # set C_ell =0 for ell=0,1 
            ucls[:,1][2:] = C_ee
            ucls[:,3][2:] = C_te
            
            C_phi, PkNL, PkL, k_out, k_out2, z_out, weyl = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, z_lens, T_path)
            print('Finished calculating non-linear lensing')
            
            lcls = correlations.lensed_cls(ucls, C_phi)
            C_tt, C_ee, C_bb, C_te = correlations.lensed_cls(ucls, C_phi).T
            #C_tt, C_ee, C_bb, C_te = ucls.T

            matter_mg3, k_index3, k_index3b = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, np.array([z1,]), T_path+str(len(z_lens) + 1), return_matter_power=True)
            matter_mg4, k_index4, k_index4b = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, np.array([z2,]), T_path+str(len(z_lens) + 2), return_matter_power=True)
            print('Finished calculating non-linear matter power')

            ## CLEAN UP FILES ## 
            os.system('rm -r '+pre_name+'_'+'scalCls.dat')
            for j in range(len(z_lens) + 2): #AL modif
                os.system('rm -r '+pre_name+'_'+'matterpower_'+str(j+1)+'.dat')
                transfer_fname = pre_name+'_'+'transfer_'+str(j+1)+'.dat'
                print('Removing transfer files for', transfer_fname)
                os.system('rm -r '+pre_name+'_'+'transfer_'+str(j+1)+'.dat')
                print('Finished removing transfer files for', transfer_fname)

            ## APPEND OUTPUT TO COLLECTIONS ##
            collection['l_index'].append(l_index)
            collection['C_tt'].append(C_tt[2:])
            collection['C_ee'].append(C_ee[2:])
            collection['C_te'].append(C_te[2:])
            collection['C_bb'].append(C_bb[2:])
            collection_1['k_index'].append(k_index)
            collection_1['matter_mg'].append(matter_mg)
            collection_2['k_index'].append(k_index2)
            collection_2['matter_mg'].append(matter_mg2)
            collection_3['k_index'].append(k_index3)
            collection_3['k_indexb'].append(k_index3b)
            collection_3['matter_mg'].append(matter_mg3)
            collection_4['k_index'].append(k_index4)
            collection_4['k_indexb'].append(k_index4b)
            collection_4['matter_mg'].append(matter_mg4)
            collection['C_phi'].append(C_phi[2:])

            #Test arrays
            collection['PkNL'].append(PkNL)
            collection['PkL'].append(PkL)
            collection['k_out'].append(k_out)
            collection['k_out2'].append(k_out2)
            collection['z_out'].append(z_out)
            collection['weyl'].append(weyl)

            for key in params:
                t_params[key].append(params[key][i])
                t_params1[key].append(params[key][i])
                t_params3[key].append(params[key][i])
            for key in ['omega_b', 'omega_cdm', 'tau_reio', 'H_0','n_s', 'A_s', 'ma', 'omega_ax', 'gamma_1', 'gamma_2']:
                t_params2[key].append(params[key][i])
                t_params4[key].append(params[key][i])
            t_params2['z'].append(z2)
            t_params4['z'].append(z2)
            print('Good!')
        except ValueError:
            os.system('rm ' + params_ini_file + '_copy')
            os.system('rm '+pre_name+'_'+'params.ini')
            s += 1
            for key in params:
                problem_list[key].append(params[key][i])
            print('problem ValueError ocurrs!')
            pass 
        except Exception:
            os.system('rm ' + params_ini_file + '_copy')
            os.system('rm '+pre_name+'_'+'params.ini')
            r += 1
            for key in params:
                problem_list_r[key].append(params[key][i])
            print('problem Exception ocurrs!')
            pass

    end = time() # save end time
    diff = end-start # elapsed time (in seconds)
    collection['time'] = diff
    collection['problem_list'] =problem_list
    collection['problem_list_r'] = problem_list_r
    collection['params'] = t_params
    collection_1['params'] = t_params1
    collection_2['params'] = t_params2
    collection_3['params'] = t_params3
    collection_4['params'] = t_params4

    ## set up index of .pkl file ##
    index_pkl = str(index_pkl_name)
    ## Finish setting up ##
    data_pkl = 'test_data_collect_HMcode_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection,f)
    f.close()
    data_pkl = 'test_data_collect_mg1_HMcode_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_1,f)
    f.close()
    data_pkl = 'test_data_collect_mg2_HMcode_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_2,f)
    f.close()

    data_pkl = 'test_data_collect_mg3_HMcode_NL_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_3,f)
    f.close()

    data_pkl = 'test_data_collect_mg4_HMcode_NL_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_4,f)
    f.close()

    print('time cost is', collection['time']) 
    print([s,r,problem_list,problem_list_r])


if __name__ == '__main__':
    inputs_list = []
    number_cores = 3 #120 # number of cores you want to use in collecting data
    for i in range(number_cores):
        pkl_name = 'LHD_parameters_6testmab_HMcode'+str(i)+'.pkl'
        outputs_name = '12params_6testmab_HMcode_' + str(i)
        os_name = 'test_os_6testmab_HMcode_' + str(i)
        pre_name = 'test_6testmab_HMcode_' + str(i) + '/test_6testmab_HMcode_' + str(i) + '_'
        ele = (pkl_name, outputs_name, os_name, pre_name)
        inputs_list.append(ele)
    start_time = time()
    p = Pool(number_cores)
    p.map(data_collection,inputs_list)
   # data_collection(('LHD_parameters_2e5_0.pkl','9params_0','test_os_0','test_0_'))
    p.close()
    p.join()
    end_time = time()
    print(end_time-start_time)
