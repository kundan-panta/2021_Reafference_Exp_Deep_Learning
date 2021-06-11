# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from correct_biases import correct_biases

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.05.25/filtered_a5_s10_o60/'  # include trailing slash
file_names_offset = 2  # difference in between actual distance and file names
# file_names = ['0', '3', '6', '6_2', '9', '12', '15', '18', '21', '24']
# file_distances = np.array([0, 3, 6, 6, 9, 12, 15, 18, 21, 24], dtype=float) + file_names_offset
file_names = ['3-2', '3-3']  # , '12-4', '12-5', '12-6', '12-7', '12-8', '12-9', '12-10', '12-11']
file_distances = np.array([3, 3], dtype=float) + file_names_offset
trajectory_name = '30deg'  # choose trajectory name for which to process data

empirical_prediction = True  # whether to use collected data as the "perfect prediction"
empirical_prediction_name = '24-1'
subract_prediction = False  # meas - pred?

save_folder = root_folder + 'plots/2021.06.11_dataplots/'  # include trailing slash
save_filename = '3_f5,10,60_'  # include traliing underscore

plot_size = (18, 12)
subplot_grid_shape = [4, 3]
idx_start = 0
idx_end = 2000

# %%
# all files to extract the data from (collected at multiple locations)
N_files = len(file_names)
assert len(file_distances) == N_files

# %%
# figure includes time series data for all locations
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({"savefig.facecolor": (1.0, 1.0, 1.0, 1)})  # disable transparent background
plt.rcParams["figure.figsize"] = plot_size
plt.tight_layout()

if empirical_prediction:  # furthest distance from wall as forward model
    ft_pred = np.loadtxt(root_folder + data_folder + empirical_prediction_name + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    if not(empirical_prediction):  # use QS model if empirical prediction is not used
        ft_pred = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

    if subract_prediction:  # subtract pred from meas?
        ft_meas -= ft_pred

    ####### normalize?? #########
    # ft_pred /= np.max(ft_pred, axis=1, keepdims=True)  # divide by max value in each row
    # ft_meas /= np.max(ft_meas, axis=1, keepdims=True)  # divide by max value in each row

    ####### normalize?? (hankun method) #########
    # ft_pred = (ft_pred - np.min(ft_pred, axis=1, keepdims=True)) / (np.max(ft_pred, axis=1, keepdims=True) - np.min(ft_pred, axis=1, keepdims=True))
    # ft_meas = (ft_meas - np.min(ft_meas, axis=1, keepdims=True)) / (np.max(ft_meas, axis=1, keepdims=True) - np.min(ft_meas, axis=1, keepdims=True))

    # a subplot for each location
    plt.figure(1)
    if k == 0:  # make axis limits the same
        ax1 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax1, sharey=ax1)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Force X (Measured) (N)')
    plt.plot(t[idx_start:idx_end], ft_meas[0, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*0.01, ':')  # superpose with stroke angle

    plt.figure(2)
    if k == 0:  # make axis limits the same
        ax2 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax2, sharey=ax2)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Force Y (Measured) (N)')
    plt.plot(t[idx_start:idx_end], ft_meas[1, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*0.001, ':')  # superpose with stroke angle

    plt.figure(3)
    if k == 0:  # make axis limits the same
        ax3 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax3, sharey=ax3)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Force Z (Measured) (N)')
    plt.plot(t[idx_start:idx_end], ft_meas[2, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*0.001, ':')  # superpose with stroke angle

    plt.figure(4)
    if k == 0:  # make axis limits the same
        ax4 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax4, sharey=ax4)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Torque X (Measured) (N-mm)')
    plt.plot(t[idx_start:idx_end], ft_meas[3, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*0.25, ':')  # superpose with stroke angle

    plt.figure(5)
    if k == 0:  # make axis limits the same
        ax5 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax5, sharey=ax5)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Torque Y (Measured) (N-mm)')
    plt.plot(t[idx_start:idx_end], ft_meas[4, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*0.5, ':')  # superpose with stroke angle

    plt.figure(6)
    if k == 0:  # make axis limits the same
        ax6 = plt.subplot(*subplot_grid_shape, k+1)
    else:
        plt.subplot(*subplot_grid_shape, k+1, sharex=ax6, sharey=ax6)
    plt.xlabel('Time (s) (' + str(file_distances[k]) + ' cm)')
    plt.ylabel('Torque Z (Measured) (N-mm)')
    plt.plot(t[idx_start:idx_end], ft_meas[5, idx_start:idx_end])
    plt.plot(t[idx_start:idx_end], ang_meas[0, idx_start:idx_end]*1, ':')  # superpose with stroke angle

# save
Path(save_folder).mkdir(parents=True, exist_ok=True)  # make folder
for i in range(1, 7):
    plt.figure(i)
    plt.savefig(save_folder + save_filename + str(i) + '.png')  # and change this (4)

# plt.show()

# %%
