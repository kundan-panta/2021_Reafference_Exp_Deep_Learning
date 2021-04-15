# %%
from math import floor
import numpy as np
import matplotlib.pyplot as plt
# from correct_biases import correct_biases

# %%
# all files to extract the data from (collected at multiple locations)
file_names = ['0', '3', '6', '12', '18']
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])
file_names_float += 3  # offset between ruler reading and distance from wing tip to wall

# choose trajectory name for which to process data
trajectory_name = '30deg'

# parameter to choose if biases should be corrected
# biases = True

# %%
# for each file, I need 6 components of rms ft
rms_all = np.zeros((N_files, 6))
rms_norm_all = np.zeros((N_files, 2))
rms_all_std = np.zeros_like(rms_all)
rms_norm_all_std = np.zeros_like(rms_norm_all)

############### alternative nominal FT - futhest from wall ###############
ft_pred = np.loadtxt('22/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    # ft_pred = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    # if biases:
    #     ft_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_bias.csv', delimiter=',', unpack=True)
    #     ang_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_bias.csv', delimiter=',', unpack=True)
    #     gravity_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'gravity_bias.csv', delimiter=',', unpack=True)

    #     # remove the three biases and rotate the frame to align with normally used frame
    #     ft_meas = correct_biases(ft_meas, ft_bias[:, 0], ang_bias[0], gravity_bias[:, 0])

    N = len(t)  # number of data points

    # find points where a new stroke cycle is started
    t_s = round(t[1] - t[0], 3)  # sample time
    freq = cpg_param[-1, 0]  # store frequencies of each param set
    t_cycle = 1 / freq  # stroke cycle time

    # calculate number of cycles
    t_total = t[-1]  # period of time over which data has been collected for each param set
    t_total += t_s  # including first point
    t_total = np.around(t_total, decimals=3)

    # calculate number of data points per cycle
    N_per_cycle = round(t_cycle / t_s)

    print('Number of data points in a cycle:')
    print(N_per_cycle)

    # N_cycles = 50
    N_cycles = floor(N / N_per_cycle)  # floor(total data points / data points in a cycle)
    print('Number of stroke cycles:')
    print(N_cycles)

    # print number of unused data points
    print('Number of unused data points:')
    print(N - N_per_cycle * N_cycles)  # total # of data points - # of data points used

    # collect rms data for each cycle -- may not be necessary
    rms_cycle = np.zeros((N_cycles, 6))
    rms_norm_cycle = np.zeros((N_cycles, 2))

    ####### normalize?? #########
    # ft_pred /= np.max(abs(ft_pred), axis=1, keepdims=True)  # divide by max value in each row
    # ft_meas /= np.max(abs(ft_meas), axis=1, keepdims=True)  # divide by max value in each row

    ####### normalize?? (hankun method) #########
    # ft_pred = (ft_pred - np.min(abs(ft_pred), axis=1, keepdims=True)) / (np.max(abs(ft_pred), axis=1, keepdims=True) - np.min(abs(ft_pred), axis=1, keepdims=True))
    # ft_meas = (ft_meas - np.min(abs(ft_meas), axis=1, keepdims=True)) / (np.max(abs(ft_meas), axis=1, keepdims=True) - np.min(abs(ft_meas), axis=1, keepdims=True))

    ####### take difference?? #########
    ft_meas -= ft_pred

    # calculate RMS values for each FT, for each stroke cycle, for each param set
    for j in range(N_cycles):
        # get ft_meas_cycle
        ft_meas_cycle = ft_meas[:, (j*N_per_cycle):((j+1)*N_per_cycle)]

        # take norm of F and T separately
        f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[0:3, :], axis=0)
        # f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[[0, 2], :], axis=0)  # only x and z forces
        T_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[3:6, :], axis=0)

        # rms
        rms_cycle[j, :] = np.sqrt(1/N_per_cycle * np.sum(ft_meas_cycle**2, axis=1))
        rms_norm_cycle[j, 0] = np.sqrt(1/N_per_cycle * np.sum(f_meas_norm_cycle**2))
        rms_norm_cycle[j, 1] = np.sqrt(1/N_per_cycle * np.sum(T_meas_norm_cycle**2))

    # average the FT RMS for across all stroke cycles
    rms_all[k, :] = np.mean(rms_cycle, axis=0)
    rms_norm_all[k, :] = np.mean(rms_norm_cycle, axis=0)

    # std dev
    rms_all_std[k, :] = np.std(rms_cycle, axis=0)
    rms_norm_all_std[k, :] = np.std(rms_norm_cycle, axis=0)

# %% subplots
plt.rcParams.update({"savefig.facecolor": (1.0, 1.0, 1.0, 1)})  # disable transparent background

plt.figure(figsize=(18, 9))

for i in range(3):  # forces
    plt.subplot(2, 3, i+1)
    plt.xlabel('Distances of wing tip from wall (cm)')
    plt.ylabel('Force ' + chr(ord('X') + i) + ' (N)')
    plt.plot(file_names_float, rms_all[:, i])
    plt.errorbar(file_names_float, rms_all[:, i], yerr=2*rms_all_std[:, i], ecolor='red', capsize=5, fmt='none')


for i in range(3):  # torques
    plt.subplot(2, 3, 3+(i+1))
    plt.xlabel('Distances of wing tip from wall (cm)')
    plt.ylabel('Torque ' + chr(ord('X') + i) + ' (N-mm)')
    plt.plot(file_names_float, rms_all[:, i+3])
    plt.errorbar(file_names_float, rms_all[:, i+3], yerr=2*rms_all_std[:, i+3], ecolor='red', capsize=5, fmt='none')

plt.savefig('plots/2021.04.11/' + trajectory_name + '/rms_difference.png')  # change this
# plt.show()

# norm
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.xlabel('Distances of wing tip from wall (cm)')
plt.ylabel('Force (Combined) (N)')
plt.plot(file_names_float, rms_norm_all[:, 0])
plt.errorbar(file_names_float, rms_norm_all[:, 0], yerr=2*rms_norm_all_std[:, 0], ecolor='red', capsize=5, fmt='none')

plt.subplot(1, 2, 2)
plt.xlabel('Distances of wing tip from wall (cm)')
plt.ylabel('Torque (Combined) (N-mm)')
plt.plot(file_names_float, rms_norm_all[:, 1])
plt.errorbar(file_names_float, rms_norm_all[:, 1], yerr=2*rms_norm_all_std[:, 1], ecolor='red', capsize=5, fmt='none')

plt.savefig('plots/2021.04.11/' + trajectory_name + '/rms_magnitude_difference.png')  # change this
# plt.show()

# %% separate plots for all ft
# for i in range(3):  # forces
#     plt.figure()
#     plt.xlabel('Distances from wall (cm)')
#     plt.ylabel('Force ' + str(i+1) + ' (N)')
#     plt.plot(file_names_float, rms_all[:, i])

# for i in range(3):  # torques
#     plt.figure()
#     plt.xlabel('Distances from wall (cm)')
#     plt.ylabel('Torque ' + str(i+1) + ' (N-mm)')
#     plt.plot(file_names_float, rms_all[:, i+3])

# # norm
# plt.figure()
# plt.xlabel('Distances from wall (cm)')
# plt.ylabel('Force (Combined) (N)')
# plt.plot(file_names_float, rms_norm_all[:, 0])

# plt.figure()
# plt.xlabel('Distances from wall (cm)')
# plt.ylabel('Torque (Combined) (N-mm)')
# plt.plot(file_names_float, rms_norm_all[:, 1])

# %%
