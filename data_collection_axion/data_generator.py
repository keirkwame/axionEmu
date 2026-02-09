
import os
import copy as cp
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
    collection_deriv={} # rh added derived params
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

    # rh added derived params
    collection_deriv['100*theta_s']=[]
    collection_deriv['sigma8']=[]
    collection_deriv['YHe']=[]
    collection_deriv['z_reio']=[]
    collection_deriv['Neff']=[]
    collection_deriv['tau_rec']=[]
    collection_deriv['z_rec']=[]
    collection_deriv['rs_rec']=[]
    collection_deriv['ra_rec']=[]
    collection_deriv['tau_star']=[]
    collection_deriv['z_star']=[]
    collection_deriv['rs_star']=[]
    collection_deriv['ra_star']=[]
    collection_deriv['rs_drag']=[]
    collection_deriv['age']=[]
    collection_deriv['kd_star']=[]
    collection_deriv['z_drag']=[]
    collection_deriv['100*theta_d']=[]

    collection_deriv['zBAO']=[]
    collection_deriv['HBAO']=[]
    collection_deriv['DABAO']=[]
    collection_deriv['rsBAO']=[]

    t_params = dict()
    t_params1 = dict()
    t_params2 = dict()
    t_params3 = dict()
    t_params4 = dict()
    deriv_params=dict() # rh added derived params
    
    # AL add z_lens generated from comoving distance
    z_lens = cp.deepcopy(np.loadtxt('../nonlinear/optimal_z_array.dat'))
    #z_lens = np.concatenate((z_lens[z_lens <= 4.][::2], z_lens[z_lens > 4.][::4]))
    print('z_lens =', z_lens)

    for key in params:
        t_params[key] = []
        t_params1[key]=[]
        t_params2[key] = []
        t_params3[key]=[]
        t_params4[key] = []
        deriv_params[key]=[] # rh added derived params
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
        #print('gamma_1, gamma_2 =', gamma_1, gamma_2)
        pre_name = pre_index + str(i) 
        try:
            print("############# " + params_ini_file)
            f1 = open(params_ini_file, 'r') #CHANGE
            lines = f1.readlines()
            f1.close()
            print('Opened and closed params.ini')

            # for l,line in enumerate(lines):
            #     print(l, line)

            # print('************')

            lines[3] = 'output_root = ' + pre_name + '\n'
            lines[36] = 'ombh2 = '+str(omega_b)+'\n' #-2 for all below # rh added derived params
            lines[37] = 'omch2 = '+str(omega_cdm)+'\n'
            lines[40] = 'hubble         = '+str(H_0)+'\n'
            lines[96] = 'scalar_amp(1)             = '+str(A_s)+'\n'
            lines[97] = 'scalar_spectral_index(1)  = '+str(n_s)+'\n'
            lines[111] = 're_optical_depth     = '+str(tau_reio)+'\n'
            lines[54] = 'omaxh2 = '+str(omega_ax)+'\n'
            lines[50] = 'm_ax = '+str(ma)+'\n'

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
            #print('New indices for Latin hypercube redshifts', z1_idx, z2_idx)

            z_combined = np.concatenate((z_lens[::-1], np.array([z1, z2])))
            z_combined_indices = np.argsort(z_combined)[::-1]
            z_ordered = z_combined[z_combined_indices]
            #print('Re-ordered and combined redshifts', z_combined_indices, z_ordered)

            lines[156] = 'transfer_num_redshifts  = '+str(len(z_lens) + 2)+'\n'
            #lines[150] = 'transfer_redshift('+ str(z1_idx) +')    = '+str(z1)+'\n'
            #lines[152] = 'transfer_redshift('+ str(z2_idx) +')    = '+str(z2)+'\n'
            '''lines[151] = 'transfer_filename('+ str(z1_idx) +')    = transfer_'+str(len(z_lens) + 1)+'.dat\n'
            lines[153] = 'transfer_filename('+ str(z2_idx) +')    = transfer_'+str(len(z_lens) + 2)+'.dat\n'
            lines[155] = 'transfer_matterpower('+ str(z1_idx) +')    = matterpower_'+str(len(z_lens) + 1)+'.dat\n'
            lines[156] = 'transfer_matterpower('+ str(z2_idx) +')    = matterpower_'+str(len(z_lens) + 2)+'.dat\n'
            '''
            lines[158] = '\n'
            lines[159] = '\n'
            lines[160] = '\n'
            lines[161] = '\n'
            lines[162] = '\n'
            lines[163] = '\n'
            lines[164] = '\n'
            lines[165] = '\n'
            lines[166] = '\n'
            
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
            os.system('/home/keir/Software/axionEmu/data_collection_axion/camb '+params_ini_file + '_copy') 
            print('Finished running CAMB after', time() - start, 'seconds')
            
            ## COLLECT DATA ##
            unlensed_cls =  np.loadtxt(pre_name+'_'+'scalCls.dat')
            l_index = unlensed_cls[:,0]
            k_index, matter_mg = np.loadtxt(pre_name+'_'+'matterpower_' + str(len(z_lens) + 1) + '.dat', unpack = True) #Linear matter power
            k_index2, matter_mg2 = np.loadtxt(pre_name+'_'+'matterpower_' + str(len(z_lens) + 2) +'.dat', unpack = True)
            print('Finished loading power spectra')

            ## NEED TO INSERT READING THE PARAMS OUTFILE HERE # rh added derived params
            # print(pre_name+'_deriv_params', '*****')
            '''df=open(pre_name+'_deriv_params')
            dlines=df.readlines()
            df.close()

            # for l, line in enumerate(dlines):
            #     print(line.split(), l)
            # print('------')
            # for key in collection_deriv:
            #     print(key)

            collection_deriv['100*theta_s'].append(float(dlines[17].split()[2]))
            collection_deriv['sigma8'].append(float(dlines[2].split()[2]))
            collection_deriv['YHe'].append(float(dlines[6].split()[2]))
            collection_deriv['z_reio'].append(float(dlines[0].split()[2]))
            collection_deriv['Neff'].append(float(dlines[1].split()[2]))
            collection_deriv['tau_rec'].append(float(dlines[4].split()[2]))
            collection_deriv['z_rec'].append(float(dlines[5].split()[2]))
            collection_deriv['rs_rec'].append(float(dlines[7].split()[2]))
            collection_deriv['ra_rec'].append(float(dlines[8].split()[2]))
            collection_deriv['tau_star'].append(float(dlines[9].split()[2]))
            collection_deriv['z_star'].append(float(dlines[10].split()[2]))
            collection_deriv['rs_star'].append(float(dlines[11].split()[2]))
            collection_deriv['ra_star'].append(float(dlines[12].split()[2]))
            collection_deriv['rs_drag'].append(float(dlines[13].split()[2]))
            collection_deriv['age'].append(float(dlines[14].split()[4]))
            collection_deriv['kd_star'].append(float(dlines[20].split()[3]))
            collection_deriv['z_drag'].append(float(dlines[18].split()[2]))
            collection_deriv['100*theta_d'].append(float(dlines[21].split()[2]))
            
            baos = np.loadtxt(pre_name+'_deriv_params_BAO', skiprows=1)
            collection_deriv['zBAO'].append(baos[:,0])
            collection_deriv['HBAO'].append(baos[:,1])
            collection_deriv['DABAO'].append(baos[:,2])
            collection_deriv['rsBAO'].append(baos[:,3])
            print('Finished loading derived parameters')
            '''

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
            
            C_phi, PkNL, PkL, k_out, z_out, weyl = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, z_lens, T_path)
            print('Finished calculating non-linear lensing')
            
            C_tt, C_ee, C_bb, C_te = correlations.lensed_cls(ucls, C_phi).T
            #C_tt, C_ee, C_bb, C_te = ucls.T

            #Non-linear matter power
            matter_mg3, k_index3 = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, np.array([z1,]), T_path+str(len(z_lens) + 1), return_matter_power=True)
            matter_mg4, k_index4 = do_non_linear_lensing(H_0, omega_cdm, omega_b, A_s, n_s, ma, omega_ax, gamma_1, gamma_2, np.array([z2,]), T_path+str(len(z_lens) + 2), return_matter_power=True)
            print('Finished calculating non-linear matter power')

            ## NEED TO INSERT READING THE PARAMS OUTFILE HERE # rh added derived params
            # print(pre_name+'_deriv_params', '*****')
            df=open(pre_name+'_deriv_params')
            dlines=df.readlines()
            df.close()

            # for l, line in enumerate(dlines):
            #     print(line.split(), l)
            # print('------')
            # for key in collection_deriv:
            #     print(key)

            collection_deriv['100*theta_s'].append(float(dlines[17].split()[2]))
            collection_deriv['sigma8'].append(float(dlines[2].split()[2]))
            collection_deriv['YHe'].append(float(dlines[6].split()[2]))
            collection_deriv['z_reio'].append(float(dlines[0].split()[2]))
            collection_deriv['Neff'].append(float(dlines[1].split()[2]))
            collection_deriv['tau_rec'].append(float(dlines[4].split()[2]))
            collection_deriv['z_rec'].append(float(dlines[5].split()[2]))
            collection_deriv['rs_rec'].append(float(dlines[7].split()[2]))
            collection_deriv['ra_rec'].append(float(dlines[8].split()[2]))
            collection_deriv['tau_star'].append(float(dlines[9].split()[2]))
            collection_deriv['z_star'].append(float(dlines[10].split()[2]))
            collection_deriv['rs_star'].append(float(dlines[11].split()[2]))
            collection_deriv['ra_star'].append(float(dlines[12].split()[2]))
            collection_deriv['rs_drag'].append(float(dlines[13].split()[2]))
            collection_deriv['age'].append(float(dlines[14].split()[4]))
            collection_deriv['kd_star'].append(float(dlines[20].split()[3]))
            collection_deriv['z_drag'].append(float(dlines[18].split()[2]))
            collection_deriv['100*theta_d'].append(float(dlines[21].split()[2]))

            baos = np.loadtxt(pre_name+'_deriv_params_BAO', skiprows=1)
            collection_deriv['zBAO'].append(baos[:,0])
            collection_deriv['HBAO'].append(baos[:,1])
            collection_deriv['DABAO'].append(baos[:,2])
            collection_deriv['rsBAO'].append(baos[:,3])
            print('Finished loading derived parameters')

            ## CLEAN UP FILES ## 
            os.system('rm -r '+pre_name+'_'+'scalCls.dat')
            os.system('rm -r '+pre_name+'_'+'deriv_params')
            os.system('rm -r '+pre_name+'_'+'deriv_params_BAO')
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
            #collection_3['k_indexb'].append(k_index3b)
            collection_3['matter_mg'].append(matter_mg3)
            collection_4['k_index'].append(k_index4)
            #collection_4['k_indexb'].append(k_index4b)
            collection_4['matter_mg'].append(matter_mg4)
            collection['C_phi'].append(C_phi[2:])

            #Test arrays
            '''collection['PkNL'].append(PkNL)
            collection['PkL'].append(PkL)
            collection['k_out'].append(k_out)
            collection['z_out'].append(z_out)
            collection['weyl'].append(weyl)
            '''

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
    collection_deriv['params'] = t_params

    ## set up index of .pkl file ##
    index_pkl = str(index_pkl_name)
    ## Finish setting up ##
    data_pkl = '/home/keir/keir/data_C_ell_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection,f)
    f.close()
    data_pkl = '/home/keir/keir/data_P_k_linear_1_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_1,f)
    f.close()
    data_pkl = '/home/keir/keir/data_P_k_linear_2_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_2,f)
    f.close()

    # rh added derived params
    data_pkl = '/home/keir/keir/data_derived_params_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_deriv,f)
    f.close()

    data_pkl = '/home/keir/keir/data_P_k_nonlinear_1_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_3,f)
    f.close()

    data_pkl = '/home/keir/keir/data_P_k_nonlinear_2_'+index_pkl+'.pkl'
    print('Dump data to '+data_pkl+'...')
    f = open(data_pkl,'wb')
    pickle.dump(collection_4,f)
    f.close()

    print('time cost is', collection['time']) 
    print([s,r,problem_list,problem_list_r])


if __name__ == '__main__':
    inputs_list = []
    number_cores = 5 # number of cores you want to use in collecting data
    root  = 'LH_ACT_DR6_TTTEEEPP_5high_axion_'
    for i in range(number_cores):
        pkl_name = '/home/keir/keir/'+root+str(i)+'.pkl'
        outputs_name = root + str(i)
        os_name = '/home/keir/keir/'+root + 'inifile_' + str(i)
        pre_name = '/home/keir/keir/'+root + str(i) + '/'+root + str(i) + '_'
        ele = (pkl_name, outputs_name, os_name, pre_name)
        inputs_list.append(ele)
    start_time = time()
    p = Pool(number_cores)
    p.map(data_collection,inputs_list)
    #data_collection(('/home/renee/axionEmuDat/LHD_parameters_NL_200k_HMcode_fixh2_0.pkl','LHD_parameters_NL_200k_HMcode_fixh2_0','test_osr_0','test_0_'))
    p.close()
    p.join()
    end_time = time()
    print(end_time-start_time)

