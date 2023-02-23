# In[ ]:


import os
import numpy as np
import pickle
import copy
from time import time
from multiprocessing import Pool

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
    collection['l_index'] = []
    collection['C_tt'] = []
    collection['C_ee'] = []
    collection['C_te'] = []
    collection['C_phi'] = []
    collection_1['k_index'] = []
    collection_1['matter_mg'] = []
    collection_2['k_index'] = []
    collection_2['matter_mg'] = []

    # AL modif (make space for NL Pk)
    collection_3['k_index'] = []
    collection_3['matter_mg'] = []
    collection_4['k_index'] = []
    collection_4['matter_mg'] = []

    t_params = dict()
    t_params1 = dict()
    t_params2 = dict()
    t_params3 = dict()
    t_params4 = dict()
    
    # AL add z_lens generated from comoving distance
    z_lens = np.loadtxt('optimal_z_array.dat')

    for key in params:
        t_params[key] = []
        t_params1[key]=[]
        t_params2[key] = []
        t_params3[key]=[]
        t_params4[key] = []
#     var_param = ["ombh2", "omch2", "re_optical_depth    ", "hubble        ", "scalar_spectral_index(1) ", "scalar_amp(1)            "]
    params_keys = ['omega_b', 'omega_cdm', 'tau_reio', 'H_0','n_s', 'A_s', 'ma', 'omega_ax', 'z']
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
        pre_name = pre_index + str(i) 
        try:
            f1 = open(params_ini_file, 'r') #CHANGE
            lines = f1.readlines()
            f1.close()
            lines[3] = 'output_root = ' + pre_name + '\n'
            lines[34] = 'ombh2 = '+str(omega_b)+'\n'
            lines[35] = 'omch2 = '+str(omega_cdm)+'\n'
            lines[38] = 'hubble         = '+str(H_0)+'\n'
            lines[91] = 'scalar_amp(1)             = '+str(A_s)+'\n'
            lines[92] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
            lines[106] = 're_optical_depth     = '+str(tau_reio)+'\n'
            lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
            lines[55] = 'm_ax = '+str(ma)+'\n'
            lines[150] = 'transfer_redshift(1)    = '+str(z1)+'\n'
            lines[152] = 'transfer_redshift(2)    = '+str(z2)+'\n'

            for i in range(len(z_lens)): #AL modif add Nz transfer calc
                lines.append('transfer_redshift('+str(i+3)+')    = '+str(z_lens[i])+'\n')
            
            os.system('rm '+params_ini_file)
            os.system('touch '+params_ini_file)
            f2 = open(params_ini_file, 'w')
            f2.writelines(lines)
            f2.close()

            ## RUN CAMB ##
            os.system('./camb '+params_ini_file)
            
            ## COLLECT DATA ##
            #l_index, C_tt, C_ee, C_te = np.loadtxt(pre_name+'_'+'scalCls.dat', unpack = True) # now doing unlensed
            unlensed_cls =  np.loadtxt(pre_name+'_'+'scalCls.dat')
            k_index, matter_mg = np.loadtxt(pre_name+'_'+'matterpower1.dat', unpack = True)
            k_index2, matter_mg2 = np.loadtxt(pre_name+'_'+'matterpower2.dat', unpack = True)
            #C_phi = np.loadtxt(pre_name+'_'+'scalCls.dat', unpack = True, usecols = 4)
            
            ## CLEAN UP FILES ##
            os.system('rm '+pre_name+'_'+'lensedCls.dat')
            os.system('rm '+pre_name+'_'+'matterpower1.dat')
            os.system('rm '+pre_name+'_'+'matterpower2.dat')
            os.system('rm '+pre_name+'_'+'scalCls.dat')
            os.system('rm '+pre_name+'_'+'transfer_out1.dat')
            os.system('rm '+pre_name+'_'+'transfer_out2.dat')
            for i in range(len(z_lens)): #AL modif
                os.system('rm '+pre_name+'_'+'matterpower'+str(i+3)'.dat')
                os.system('rm '+pre_name+'_'+'transfer_out'+str(i+3)+'.dat')
            os.system('rm '+pre_name+'_'+'scalCovCls.dat')
            os.system('rm '+pre_name+'_'+'lenspotentialCls.dat')
            os.system('rm '+pre_name+'_'+'params.ini')
            
            ## PERFORM NON-LINEAR TRANSFORMS ##
            T_path = pre_name+'_'+'transfer_out'
            C_tt, C_ee, C_te, C_phi = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, m_ax, omega_ax, z_lens, T_path, unlensed_cls)
            matter_mg3 = do_non_linear_pk(H_0, omega_cdm, omega_b, A_s, n_s, m_ax, omega_ax, z1, T_path+'1')
            matter_mg4 = do_non_linear_pk(H_0, omega_cdm, omega_b, A_s, n_s, m_ax, omega_ax, z2, T_path+'2')

            ## APPEND OUTPUT TO COLLECTIONS ##
            collection['l_index'].append(l_index)
            collection['C_tt'].append(C_tt)
            collection['C_ee'].append(C_ee)
            collection['C_te'].append(C_te)
            collection_1['k_index'].append(k_index)
            collection_1['matter_mg'].append(matter_mg)
            collection_2['k_index'].append(k_index2)
            collection_2['matter_mg'].append(matter_mg2)
            collection_3['k_index'].append(k_index3)
            collection_3['matter_mg'].append(matter_mg3)
            collection_4['k_index'].append(k_index4)
            collection_4['matter_mg'].append(matter_mg4)
            collection['C_phi'].append(C_phi)

            for key in params:
                t_params[key].append(params[key][i])
                t_params1[key].append(params[key][i])
            for key in ['omega_b', 'omega_cdm', 'tau_reio', 'H_0','n_s', 'A_s', 'ma', 'omega_ax']:
                t_params2[key].append(params[key][i])
            t_params2['z'].append(z2)
            print('Good!')
        except ValueError:
            os.system('rm '+pre_name+'_'+'params.ini')
            s += 1
            for key in params:
                problem_list[key].append(params[key][i])
            print('problem ValueError ocurrs!')
            pass 
        except Exception:
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
    data_pkl = 'test_data_collect_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection,f)
    f.close()
    data_pkl = 'test_data_collect_mg1_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_1,f)
    f.close()
    data_pkl = 'test_data_collect_mg2_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_2,f)
    f.close()
    print('time cost is', collection['time']) 
    print([s,r,problem_list,problem_list_r])


if __name__ == '__main__':
    inputs_list = []
    number_cores = 60 # number of cores you want to use in collecting data
    for i in range(number_cores):
        pkl_name = 'LHD_parameters_5e5_mp'+str(i)+'.pkl'
        outputs_name = '9params_5e5_mp_test_' + str(i)
        os_name = 'test_os_mp_' + str(i)
        pre_name = 'test_mp_' + str(i) + '_'
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
