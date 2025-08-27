import os
# When run this script, make sure default_inifile_00 is in the same directory -- this is the default inifile for axionCAMB

number_cores = 2 # this number should be consistent with number_cores variable in the 9parameters_data_collection_mp.py file and ideally num_subfile in LH_sampling.py
for i in range(number_cores):
    new_folder = '/home/keir/keir/LH_ACT_DR6_TTTEEEPP_derived_Planck_LCDM_gfortran_orig_fast_'+str(i)
    os.system('mkdir ' + new_folder)
    new_name = '/home/keir/keir/LH_ACT_DR6_TTTEEEPP_derived_Planck_LCDM_gfortran_orig_fast_inifile_'+str(i)
    f1=open('default_inifile_00', 'r')
    lines = f1.readlines()
    f1.close()
    lines[3] = 'output_root = '+ new_folder +'/LH_ACT_DR6_TTTEEEPP_derived_Planck_LCDM_gfortran_orig_fast_'+str(i)+'\n'
    os.system('touch '+new_name)
    f2 = open(new_name, 'w')
    f2.writelines(lines)
    f2.close()

