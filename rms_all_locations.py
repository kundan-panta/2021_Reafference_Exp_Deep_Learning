import numpy as np


#%%
# all files to extract the data from (collected at multiple locations)
file_names = ['12','14','16','18','20','22','24','26','28','30']

# choose trajectory name for which to process data
trajectory_name = '30deg'

# I might have to use a for loop (for ease) at each location, since the length of data at each location might differ by a small number


#%%
# get data
t = np.around(np.loadtxt('t.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
ft_meas = np.loadtxt('ft_meas.csv', delimiter=',', unpack=True)
ang_meas = np.loadtxt('ang_meas.csv', delimiter=',', unpack=True)
cpg_param = np.loadtxt('cpg_param.csv', delimiter=',', unpack=True)


#%%
N = len(cpg_param[1])  # number of data points

####################### WILL BE OBSOLETE ########################
## find points where new cpg parameters are used
cpg_param_change = cpg_param[:, 0:(N-1)] - cpg_param[:, 1:N]
cpg_param_change_idx = np.nonzero(cpg_param_change)[1] + 1
# [1] means take the 2nd input of the nonzero() fcn, which gives where nonzero input by column
# +1 needed so idx is for start of new param set, rather than end of old param set

# test = cpg_param[:, cpg_param_change_idx[1] - 1]  # checking if the cpg params really change at the index

cpg_param_change_idx = np.insert(cpg_param_change_idx, 0, 0)  # including the first one
cpg_param_change_idx = cpg_param_change_idx.astype(int)

N_param = len(cpg_param_change_idx)  # number of param sets
#################################################################


## find points where a new stroke cycle is started
t_s = round(t[1] - t[0], 3)  # sample time
f_param = cpg_param[-1, cpg_param_change_idx]  # store frequencies of each param set
t_cycle_param = np.around(1 / f_param, decimals=3)  # stroke cycle time

# calculate number of cycles
t_param = np.zeros(N_param)  # period of time over which data has been collected for each param set
for i in range(N_param):
    if i < N_param-1:
        t_param[i] = t[cpg_param_change_idx[i+1] - 1] - t[cpg_param_change_idx[i]]
    else:
        t_param[i] = t[-1] - t[cpg_param_change_idx[i]]
t_param += t_s  # including first point
t_param = np.around(t_param, decimals=3)

N_cycles = (t_param / t_cycle_param).astype(int)  # total time of data collection / time for 1 stroke cycle
if np.any(N_cycles != N_cycles[0]):
    print('Different number of cycles for different parameter sets.')
    raise
N_cycles = N_cycles[0]

# calculate number of data points per cycle
N_per_cycle = (t_cycle_param / t_s).astype(int)

# print number of unused data points
print('Number of unused data points:')
print((t_param / t_s).astype(int) - N_per_cycle * N_cycles)  # total # of data points - # of data points used


#%%
# 3d matrix, 1st dim is a param set, 2nd dim is stroke cycle, 3rd dim are the 6 FT
# need to have same number of cycles for each param set
# at the same time, also do the same for the magnitude of FT instead of the 3 directions separately
rms_all = np.zeros((N_param, N_cycles, 6))
rms_norm_all = np.zeros((N_param, N_cycles, 2))

## calculate RMS values for each FT, for each stroke cycle, for each param set
for i in range(N_param):
    for j in range(N_cycles):
        # get ft_meas_cycle
        ft_meas_cycle = ft_meas[:, (cpg_param_change_idx[i] + j*N_per_cycle[i]):(cpg_param_change_idx[i] + (j+1)*N_per_cycle[i])]
        # take norm of F and T separately
        f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[0:3,:],axis=0)
        T_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[3:6,:],axis=0)

        # rms
        rms_all[i, j, :] = np.sqrt(1/N_per_cycle[i] * np.sum(ft_meas_cycle**2, axis=1))
        rms_norm_all[i, j, 0] = np.sqrt(1/N_per_cycle[i] * np.sum(f_meas_norm_cycle**2))
        rms_norm_all[i, j, 1] = np.sqrt(1/N_per_cycle[i] * np.sum(T_meas_norm_cycle**2))

# average the FT RMS for across all stroke cycles
rms_avg = np.mean(rms_all, axis=1)
rms_norm_avg = np.mean(rms_norm_all, axis=1)
