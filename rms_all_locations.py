# %%
import numpy as np


# %%
# all files to extract the data from (collected at multiple locations)
file_names = ['12', '14', '16', '18', '20', '22', '24', '26', '28', '30']
N_files = len(file_names)

# choose trajectory name for which to process data
trajectory_name = '30deg'

# I might have to use a for loop (for ease) at each location, since the length of data at each location might differ by a small number

# %%
# for each file, I need 6 components of rms ft
rms_all = np.zeros((N_files, 6))
rms_norm_all = np.zeros((N_files, 2))

# %%
for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    # %%
    N = len(cpg_param[1])  # number of data points

    ####################### WILL BE OBSOLETE ########################
    # # find points where new cpg parameters are used
    # cpg_param_change = cpg_param[:, 0:(N-1)] - cpg_param[:, 1:N]
    # cpg_param_change_idx = np.nonzero(cpg_param_change)[1] + 1
    # # [1] means take the 2nd input of the nonzero() fcn, which gives where nonzero input by column
    # # +1 needed so idx is for start of new param set, rather than end of old param set

    # # test = cpg_param[:, cpg_param_change_idx[1] - 1]  # checking if the cpg params really change at the index

    # cpg_param_change_idx = np.insert(cpg_param_change_idx, 0, 0)  # including the first one
    # cpg_param_change_idx = cpg_param_change_idx.astype(int)

    # N_param = len(cpg_param_change_idx)  # number of param sets
    #################################################################

    # find points where a new stroke cycle is started
    t_s = round(t[1] - t[0], 3)  # sample time
    f_param = cpg_param[-1, 0]  # store frequencies of each param set
    t_cycle_param = np.around(1 / f_param, decimals=3)  # stroke cycle time

    # calculate number of cycles
    t_param = t[-1]  # period of time over which data has been collected for each param set
    t_param += t_s  # including first point
    t_param = np.around(t_param, decimals=3)

    N_cycles = (t_param / t_cycle_param).astype(int)  # total time of data collection / time for 1 stroke cycle
    print('Number of stroke cycles:')
    print(N_cycles)

    # calculate number of data points per cycle
    N_per_cycle = (t_cycle_param / t_s).astype(int)

    # print number of unused data points
    print('Number of unused data points:')
    print((t_param / t_s).astype(int) - N_per_cycle * N_cycles)  # total # of data points - # of data points used

    # %%
    # 3d matrix, 1st dim is a param set, 2nd dim is stroke cycle, 3rd dim are the 6 FT
    # need to have same number of cycles for each param set
    # at the same time, also do the same for the magnitude of FT instead of the 3 directions separately
    rms = np.zeros((N_cycles, 6))
    rms_norm = np.zeros((N_cycles, 2))

    # calculate RMS values for each FT, for each stroke cycle, for each param set
    for j in range(N_cycles):
        # get ft_meas_cycle
        ft_meas_cycle = ft_meas[:, (j*N_per_cycle):((j+1)*N_per_cycle)]
        # take norm of F and T separately
        f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[0:3, :], axis=0)
        T_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[3:6, :], axis=0)

        # rms
        rms[j, :] = np.sqrt(1/N_per_cycle * np.sum(ft_meas_cycle**2, axis=1))
        rms_norm[j, 0] = np.sqrt(1/N_per_cycle * np.sum(f_meas_norm_cycle**2))
        rms_norm[j, 1] = np.sqrt(1/N_per_cycle * np.sum(T_meas_norm_cycle**2))

    # average the FT RMS for across all stroke cycles
    rms_all[k, :] = np.mean(rms, axis=0)
    rms_norm_all[k, :] = np.mean(rms_norm, axis=0)
