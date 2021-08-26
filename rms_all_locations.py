# %%
# from math import floor
import numpy as np
import matplotlib.pyplot as plt
# from correct_biases import correct_biases

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/butterworth_h0.04_l5_o10/'  # include trailing slash
Ro = 3.5
A_star = 2

sets_train = [1, 2, 3, 4, 5]
d_train = [list(range(1, 43 + 1, 3))] * 5  # list of all distances from wall for each set
d_train_labels = d_train

# sets_test = [101]
# d_test = [list(range(1, 37 + 1, 3))] * 1  # list of all distances from wall
# d_test_labels = d_test

# separate_test_files = len(sets_test) > 0
# if separate_test_files:
#     train_test_split = 1
#     shuffle_examples = False
# else:
#     train_test_split = 0.8
#     shuffle_examples = True
#     shuffle_seed = 5  # seed to split data in reproducible way

# N_cycles_example = 1  # use this number of stroke cycles as 1 example
# N_cycles_step = 1  # number of cycles to step between consecutive examples
# # total number of cycles to use per file
# # set 0 to automatically calculate number of examples from the first file
# N_cycles_to_use = 0

# inputs_ft = [0, 1, 2, 3, 4, 5]
# inputs_ang = [0]

baseline_d = None  # set to None for no baseline

# lstm_units = 64  # number of lstm cells of each lstm layer
# lr = 0.0001  # learning rate
# epochs_number = 1500  # number of epochs
# epochs_patience = 300  # for early stopping, set <0 to disable

# save_model = True  # save model file, save last model if model_checkpoint == False
# model_checkpoint = False  # doesn't do anything if save_model == False
# save_results = True
save_folder = root_folder + 'plots/2021.08.25_rms/'  # include trailing slash
# # save_filename = ','.join(file_names_train) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2l' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# # save_filename = 'all_' + ','.join(str(temp) for temp in file_labels_test) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2g' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# save_filename = 'Ro={}_A={}_Tr={}_Te={}_in={}_bl={}_Nc={}_Ns={}_2L{}_lr={}'.format(
#     Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_test),
#     ','.join(str(temp) for temp in inputs_ft), baseline_d, N_cycles_example, N_cycles_step, lstm_units, lr)

# %%
# test that the sets and distances are assigned correctly
assert len(sets_train) == len(d_train)
for i in range(len(sets_train)):
    assert len(d_train[i]) == len(d_train_labels[i])

# assert len(sets_test) == len(d_test)
# for i in range(len(sets_test)):
#     assert len(d_test[i]) == len(d_test_labels[i])

# get the file names and labels
file_names_train = []
file_labels_train = []
for s_index, s in enumerate(sets_train):
    for d_index, d in enumerate(d_train[s_index]):
        file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
        file_labels_train.append(d_train_labels[s_index][d_index])

# file_names_test = []
# file_labels_test = []
# for s_index, s in enumerate(sets_test):
#     for d_index, d in enumerate(d_test[s_index]):
#         file_names_test.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
#         file_labels_test.append(d_test_labels[s_index][d_index])

file_names = file_names_train  # + file_names_test
file_labels = file_labels_train  # + file_labels_test

# baseline file names for each set
if baseline_d is not None:
    baseline_file_names_train = []
    for s_index, s in enumerate(sets_train):
        for d_index, d in enumerate(d_train[s_index]):
            baseline_file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

    # baseline_file_names_test = []
    # for s_index, s in enumerate(sets_test):
    #     for d_index, d in enumerate(d_test[s_index]):
    #         baseline_file_names_test.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

    baseline_file_names = baseline_file_names_train  # + baseline_file_names_test
    assert len(baseline_file_names) == len(file_names)

# %%
##################
N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0
##################

# N_files_train = len(file_names_train)
# N_files_test = len(file_names_test)
# if not(separate_test_files):  # if separate test files are not provided, then we use all the files for both training and testing
#     N_files_test = N_files_train
N_files_all = len(file_names)

# assert len(file_labels) == N_files_all  # makes sure labels are there for all files

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(data_folder + file_names[0] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(data_folder + file_names[0] + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

t_s = round(t[1] - t[0], 3)  # sample time
freq = cpg_param[-1, 0]  # store frequency of param set
t_cycle = 1 / freq  # stroke cycle time

N_per_cycle = round(t_cycle / t_s)  # number of data points per cycle, round instead of floor

if N_cycles_to_use == 0:  # if number of cycles per file is not explicitly specified
    N_total = len(t)  # number of data points
else:
    N_total = N_cycles_to_use * N_per_cycle + 100  # limit amount of data to use

N_per_example = round(N_cycles_example * t_cycle / t_s)  # number of data points per example, round instead of floor
N_per_step = round(N_cycles_step * t_cycle / t_s)
N_examples = (N_total - N_per_example) // N_per_step + 1  # floor division
assert N_total >= (N_examples - 1) * N_per_step + N_per_example  # last data point used must not exceed total number of data points

# number of training and testing stroke cycles
# N_examples_train = round(train_test_split * N_examples)
# if separate_test_files:
#     N_examples_test = N_examples
# else:
#     N_examples_test = N_examples - N_examples_train

# N_inputs_ft = len(inputs_ft)
# N_inputs_ang = len(inputs_ang)
# N_inputs = N_inputs_ft + N_inputs_ang  # ft_meas + other inputs

# N_classes = len(np.unique(file_labels))
# assert np.max(file_labels) == N_classes - 1  # check for missing labels in between

print('Frequency:', freq)
print('Data points in an example:', N_per_example)
print('Unused data points:', N_total - ((N_examples - 1) * N_per_step + N_per_example))  # print number of unused data points
print('Total examples per file:', N_examples)
# print('Training examples per file:', N_examples_train)
# print('Testing examples per file:', N_examples_test)
# print('Inputs:', N_inputs)
# print('Clases:', N_classes)

# making compatible with old variable names
N_cycles = N_examples

# %%
# for each file (dim 0), for each cycle (dim 1), I need 6 components of rms (dim 2)
data = np.zeros((N_files_all, N_total, 6))  # for storing all data
rms_all = np.zeros((N_files_all, N_cycles, 6))
rms_all_combined = np.zeros((N_files_all, N_cycles, 2))

for k in range(N_files_all):
    # get data
    # t = np.around(np.loadtxt(data_folder + file_names[k] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    ft_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ft_meas.csv', delimiter=',', unpack=False)
    # ang_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

    if baseline_d is not None:  # subtract pred from meas?
        baseline_ft_meas = np.loadtxt(data_folder + baseline_file_names[k] + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
        ft_meas -= baseline_ft_meas

    data[k] = ft_meas

    # N = len(t)  # number of data points

    # # find points where a new stroke cycle is started
    # t_s = round(t[1] - t[0], 3)  # sample time
    # freq = cpg_param[-1, 0]  # store frequencies of each param set
    # t_cycle = 1 / freq  # stroke cycle time

    # # calculate number of cycles
    # t_total = t[-1]  # period of time over which data has been collected for each param set
    # t_total += t_s  # including first point
    # t_total = np.around(t_total, decimals=3)

    # # calculate number of data points per cycle
    # N_per_cycle = round(t_cycle / t_s)

    # print('Number of data points in a cycle:')
    # print(N_per_cycle)

    # # N_cycles = 50
    # N_cycles = floor(N / N_per_cycle)  # floor(total data points / data points in a cycle)
    # print('Number of stroke cycles:')
    # print(N_cycles)

    # # print number of unused data points
    # print('Number of unused data points:')
    # print(N - N_per_cycle * N_cycles)  # total # of data points - # of data points used

# %% normalize
# data_min = np.reshape(np.min(np.reshape(data, (-1, 6)), axis=0), (1, -1, 6))
# data_max = np.reshape(np.max(np.reshape(data, (-1, 6)), axis=0), (1, -1, 6))

# data = (data - data_min) / (data_max - data_min)  # normalize

# %%
for k in range(N_files_all):
    ft_meas = data[k].T

    # # collect rms data for each cycle -- may not be necessary
    # rms_cycle = np.zeros((N_cycles, 6))
    # rms_norm_cycle = np.zeros((N_cycles, 2))

    # calculate RMS values for each FT, for each stroke cycle, for each param set
    for j in range(N_cycles):
        # get ft_meas_cycle
        ft_meas_cycle = ft_meas[:, (j * N_per_cycle):((j + 1) * N_per_cycle)]

        # take norm of F and T separately
        f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[0:3, :], axis=0)
        # f_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[[0, 2], :], axis=0)  # only x and z forces
        T_meas_norm_cycle = np.linalg.norm(ft_meas_cycle[3:6, :], axis=0)

        # rms
        rms_all[k, j, :] = np.sqrt(1 / N_per_cycle * np.sum(ft_meas_cycle**2, axis=1))
        rms_all_combined[k, j, 0] = np.sqrt(1 / N_per_cycle * np.sum(f_meas_norm_cycle**2))
        rms_all_combined[k, j, 1] = np.sqrt(1 / N_per_cycle * np.sum(T_meas_norm_cycle**2))

    # average the FT RMS for across all stroke cycles
    # rms_all[k, :] = np.mean(rms_cycle, axis=0)
    # rms_all_combined[k, :] = np.mean(rms_norm_cycle, axis=0)

    # std dev
    # rms_all_std[k, :] = np.std(rms_cycle, axis=0)
    # rms_combined_all_std[k, :] = np.std(rms_norm_cycle, axis=0)

# %% average for each distance as well
# mean and std are calculated based on the rms of all cycles at each distance from wall
d_all_labels = np.unique(file_labels)
rms_d_mean = np.zeros([len(d_all_labels), 6])
rms_d_std = np.zeros_like(rms_d_mean)
rms_d_mean_combined = np.zeros([len(d_all_labels), 2])
rms_d_std_combined = np.zeros_like(rms_d_mean_combined)

for d_index, d in enumerate(d_all_labels):
    # all stroke cycles for a distance, in a 2d matrix
    rms_d_all = np.reshape(rms_all[np.array(file_labels) == d], (-1, 6))
    rms_d_all_combined = np.reshape(rms_all_combined[np.array(file_labels) == d], (-1, 2))

    rms_d_mean[d_index, :] = np.mean(rms_d_all, axis=0)
    rms_d_std[d_index, :] = np.std(rms_d_all, axis=0)
    rms_d_mean_combined[d_index, :] = np.mean(rms_d_all_combined, axis=0)
    rms_d_std_combined[d_index, :] = np.std(rms_d_all_combined, axis=0)

# %% subplots
plt.rcParams.update({"savefig.facecolor": (1, 1, 1, 1)})  # disable transparent background

plt.figure(figsize=(18, 9))

for i in range(3):  # forces
    plt.subplot(2, 3, i + 1)
    plt.xlabel('Distances of wing tip from wall (cm)')
    plt.ylabel('Force ' + chr(ord('X') + i) + ' (N)')
    plt.plot(d_all_labels, rms_d_mean[:, i])
    plt.errorbar(d_all_labels, rms_d_mean[:, i], yerr=2*rms_d_std[:, i], ecolor='red', capsize=5, fmt='none')


for i in range(3, 6):  # torques
    plt.subplot(2, 3, i + 1)
    plt.xlabel('Distances of wing tip from wall (cm)')
    plt.ylabel('Torque ' + chr(ord('X') + i - 3) + ' (N-mm)')
    plt.plot(d_all_labels, rms_d_mean[:, i])
    plt.errorbar(d_all_labels, rms_d_mean[:, i], yerr=2*rms_d_std[:, i], ecolor='red', capsize=5, fmt='none')

plt.savefig(save_folder + 'rms.png')  # change this
plt.show()

# norm
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.xlabel('Distances of wing tip from wall (cm)')
plt.ylabel('Force (Combined) (N)')
plt.plot(d_all_labels, rms_d_mean_combined[:, 0])
plt.errorbar(d_all_labels, rms_d_mean_combined[:, 0], yerr=2*rms_d_std_combined[:, 0], ecolor='red', capsize=5, fmt='none')

plt.subplot(1, 2, 2)
plt.xlabel('Distances of wing tip from wall (cm)')
plt.ylabel('Torque (Combined) (N-mm)')
plt.plot(d_all_labels, rms_d_mean_combined[:, 1])
plt.errorbar(d_all_labels, rms_d_mean_combined[:, 1], yerr=2*rms_d_std_combined[:, 1], ecolor='red', capsize=5, fmt='none')

plt.savefig(save_folder + 'rms_magnitude.png')  # change this
plt.show()

# %%
