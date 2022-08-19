The 9parameters_data_collection_mp.py is used to collecting training data from axionCAMB. 
You should firstly use LDH_sampling_5e5_mp.py to generate input parameters. Then use test_os_generator.py to generate n ini files (n = number of CPU cores you want to use).
Note that number_cores varaible in 9parameters_data_collection_mp.py, num_subfile variable in LDH_sampling_5e5_mp.py, and number_cores in test_os_generator.py should be set to same values.
