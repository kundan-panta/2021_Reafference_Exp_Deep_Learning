# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# all files to extract the data from (collected at multiple locations)
file_names = ['12', '14', '16', '18', '20', '22', '24', '26', '28', '30']
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])

# choose trajectory name for which to process data
trajectory_name = '30deg'

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

    N = len(cpg_param[1])  # number of data points

    # find points where a new stroke cycle is started
    t_s = round(t[1] - t[0], 3)  # sample time
    freq = cpg_param[-1, 0]  # store frequencies of each param set
    t_cycle = np.around(1 / freq, decimals=3)  # stroke cycle time

    # calculate number of cycles
    t_total = t[-1]  # period of time over which data has been collected for each param set
    t_total += t_s  # including first point
    t_total = np.around(t_total, decimals=3)

    N_cycles = (t_total / t_cycle).astype(int)  # total time of data collection / time for 1 stroke cycle
    print('Number of stroke cycles:')
    print(N_cycles)

    # calculate number of data points per cycle
    N_per_cycle = (t_cycle / t_s).astype(int)

    print('Number of data points in a cycle:')
    print(N_per_cycle)

    # print number of unused data points
    print('Number of unused data points:')
    print((t_total / t_s).astype(int) - N_per_cycle * N_cycles)  # total # of data points - # of data points used

    # collect rms data for each cycle -- may not be necessary
    rms_cycle = np.zeros((N_cycles, 6))
    rms_norm_cycle = np.zeros((N_cycles, 2))

    # calculate RMS values for each FT, for each stroke cycle, for each param set
    for j in range(N_cycles):
        # get ft_meas_cycle
        ft_meas_cycle = ft_meas[:, (j*N_per_cycle):((j+1)*N_per_cycle)]
        # take norm of F and T separately
        f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[0:3, :], axis=0)
        T_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[3:6, :], axis=0)

        # rms
        rms_cycle[j, :] = np.sqrt(1/N_per_cycle * np.sum(ft_meas_cycle**2, axis=1))
        rms_norm_cycle[j, 0] = np.sqrt(1/N_per_cycle * np.sum(f_meas_norm_cycle**2))
        rms_norm_cycle[j, 1] = np.sqrt(1/N_per_cycle * np.sum(T_meas_norm_cycle**2))

    # average the FT RMS for across all stroke cycles
    rms_all[k, :] = np.mean(rms_cycle, axis=0)
    rms_norm_all[k, :] = np.mean(rms_norm_cycle, axis=0)

# %% separate plots
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

# %% subplots
plt.figure(figsize=(18, 9))

for i in range(3):  # forces
    plt.subplot(2, 3, i+1)
    plt.xlabel('Distances from wall (cm)')
    plt.ylabel('Force ' + str(i+1) + ' (N)')
    plt.plot(file_names_float, rms_all[:, i])

for i in range(3):  # torques
    plt.subplot(2, 3, 3+(i+1))
    plt.xlabel('Distances from wall (cm)')
    plt.ylabel('Torque ' + str(i+1) + ' (N-mm)')
    plt.plot(file_names_float, rms_all[:, i+3])

plt.show()

# norm
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.xlabel('Distances from wall (cm)')
plt.ylabel('Force (Combined) (N)')
plt.plot(file_names_float, rms_norm_all[:, 0])

plt.subplot(1, 2, 2)
plt.xlabel('Distances from wall (cm)')
plt.ylabel('Torque (Combined) (N-mm)')
plt.plot(file_names_float, rms_norm_all[:, 1])

plt.show()

# %%
