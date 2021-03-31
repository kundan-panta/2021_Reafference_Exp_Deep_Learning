# %%
import numpy as np
import matplotlib.pyplot as plt
from correct_biases import correct_biases

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

# parameter to choose if biases should be corrected
biases = True

# %%
# figure includes time series data for all locations
plt.figure(figsize=(18, 8))
plt.rcParams.update({'font.size': 8})
plt.tight_layout()

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    ft_pred = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    if biases:
        ft_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_bias.csv', delimiter=',', unpack=True)
        ang_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_bias.csv', delimiter=',', unpack=True)
        gravity_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'gravity_bias.csv', delimiter=',', unpack=True)

        # remove the three biases and rotate the frame to align with normally used frame
        ft_meas = correct_biases(ft_meas, ft_bias[:, 0], ang_bias[0], gravity_bias[:, 0])

    # a subplot for each location
    plt.subplot(3, 4, k+1)
    plt.xlabel('Time (s)')
    plt.ylabel('Force y (Meas) (N) (' + str(file_names_float[k]) + ' cm)')
    plt.plot(t[0:1000], ft_meas[1, 0:1000])
    plt.plot(t[0:1000], ang_meas[0, 0:1000]*0.25, ':')  # superpose with stroke angle

# plt.show()
plt.savefig('plots/all_fy_meas_30deg.png')

# %%
