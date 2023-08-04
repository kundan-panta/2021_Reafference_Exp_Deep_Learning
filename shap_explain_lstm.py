# %%
from helper_functions import divide_file_names, data_get_info, data_load
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_v2_behavior()

# %%
save_filename = "Ro=3.5_A=2_Tr=1,2,4,5_Te=3_in=0,1,2,3,4,5_bl=None_Nc=1_Ns=1_2L16_lr=0.0002_win=10_trun=1"
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
Ro = 3.5
A_star = 2

sets_train = [1, 2, 4, 5]
d_train = [list(range(1, 43 + 1, 3))] * 4  # list of all distances from wall for each set
d_train_labels = d_train

sets_val = [3]
d_val = [list(range(1, 43 + 1, 3))]  # list of all distances from wall
d_val_labels = d_val

separate_val_files = len(sets_val) > 0
if separate_val_files:
    train_val_split = 1
    shuffle_examples = False
    shuffle_seed = None
else:
    train_val_split = 0.8
    shuffle_examples = True
    shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = []
average_window = 10
truncate_sequence = 1

baseline_d = None  # set to None for no baseline

lstm_units = 16  # number of lstm cells of each lstm layer
lr = 0.0002  # learning rate
epochs_number = 5000  # number of epochs
epochs_patience = 500  # for early stopping, set <0 to disable

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2021.12.22_averaged/'  # include trailing slash

# %% get the file names to load data from
file_names, file_labels,\
    file_names_train, file_labels_train,\
    file_names_val, file_labels_val,\
    baseline_file_names_train, baseline_file_names_val,\
    baseline_file_names = divide_file_names(sets_train, d_train, d_train_labels,
                                            sets_val, d_val, d_val_labels,
                                            baseline_d,
                                            Ro, A_star)

# %% get info about the data
N_files_all, N_files_train, N_files_val,\
    N_examples, N_examples_train, N_examples_val,\
    N_per_example, N_per_step, N_total,\
    N_inputs, N_inputs_ft, N_inputs_ang = data_get_info(data_folder,
                                                        file_names, file_labels,
                                                        file_names_train, file_names_val,
                                                        train_val_split, separate_val_files,
                                                        N_cycles_example, N_cycles_step, N_cycles_to_use,
                                                        inputs_ft, inputs_ang)

# %% get training and validation datasets
X_train, y_train, X_val, y_val = data_load(data_folder,
                                           file_names, file_labels,
                                           baseline_d, baseline_file_names,
                                           inputs_ft, inputs_ang,
                                           separate_val_files, shuffle_examples, shuffle_seed,
                                           save_model, save_results, save_folder, save_filename,
                                           N_files_all, N_files_train,
                                           N_examples, N_examples_train, N_examples_val,
                                           N_per_example, N_per_step,
                                           N_inputs, N_inputs_ft, N_inputs_ang)

# %% reduce sequence length
N_per_example = N_per_example // average_window  # update sequence length
print('Data points in an example after averaging:', N_per_example)

# cut out last data points so the number of data points is divisible by average_window
X_train = X_train[:, 0:N_per_example * average_window, :]
X_val = X_val[:, 0:N_per_example * average_window, :]

# reshape the time series so
X_train = X_train.reshape(X_train.shape[0], -1, average_window, X_train.shape[2]).mean(axis=2)
X_val = X_val.reshape(X_val.shape[0], -1, average_window, X_val.shape[2]).mean(axis=2)

# %% truncate sequence further
N_per_example = round(N_per_example * truncate_sequence)
X_train = X_train[:, 0:N_per_example, :]
X_val = X_val[:, 0:N_per_example, :]

# %% load model
model = keras.models.load_model(save_folder + save_filename)

# %% get shap values
# select a set of background examples to take an expectation over
# background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
background = X_train

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_val)

# %% plot shap values
shap_val = shap_values
shap_val = np.array(shap_val)
shap_val = np.reshape(shap_val, (int(shap_val.shape[1]), int(shap_val.shape[2]), int(shap_val.shape[3])))
shap_abs = np.abs(shap_val)
shap_mean = np.mean(shap_abs, axis=0)
shap_std = np.std(shap_abs, axis=0)

# %% per time step
f_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
x_pos = [i for i, _ in enumerate(f_names)]

plt.figure(0)
plt1 = plt.subplot(311)
plt1.barh(x_pos, shap_mean[1])
plt1.set_yticks(x_pos)
plt1.set_yticklabels(f_names)
plt1.set_title("Yesterday's features (time-step 2)")
plt2 = plt.subplot(312, sharex=plt1)
plt2.barh(x_pos, shap_mean[0])
plt2.set_yticks(x_pos)
plt2.set_yticklabels(f_names)
plt2.set_title("The day before yesterday's features(time-step 1)")
plt.tight_layout()
plt.show()

# %% plot shap time series along with the force components
# the average forces and torques over all cycles
# TODO
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

fig = plt.figure(1, figsize=(18, 6))
axes = [0] * 6

# for i in range(3):
#     axes[i] = plt.subplot(2, 3, i + 1)
#     plt.xlabel('Time steps')
#     plt.ylabel('Force ' + chr(ord('X') + i))
#     plt.plot(shap_mean[:, i])

#     axes[3 + i] = plt.subplot(2, 3, 3 + (i + 1))
#     plt.xlabel('Time steps')
#     plt.ylabel('Torque ' + chr(ord('X') + i))
#     plt.plot(shap_mean[:, i + 3])

plt.rcParams.update({"savefig.facecolor": (1, 1, 1, 1)})  # disable transparent background
plt.rc('font', family='serif', size=12)
plt.tight_layout()

fig.supxlabel('Time steps')
gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.35)  # workaround to have no overlap between subplots

for i in range(3):  # forces
    axes[i] = plt.subplot(gs[i])
    plt.ylabel('Force ' + chr(ord('X') + i))
    plt.plot(shap_mean[:, i], 'r-')
    plt.gca().fill_between(np.arange(shap_mean.shape[0]), shap_mean[:, i] + shap_std[:, i], shap_mean[:, i] - shap_std[:, i], color="#dddddd")
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

for i in range(3, 6):  # torques
    axes[i] = plt.subplot(gs[i])
    plt.ylabel('Torque ' + chr(ord('X') + i - 3))
    plt.plot(shap_mean[:, i], 'r-')
    plt.gca().fill_between(np.arange(shap_mean.shape[0]), shap_mean[:, i] + shap_std[:, i], shap_mean[:, i] - shap_std[:, i], color="#dddddd")
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

for i in range(5):
    axes[i + 1].sharey(axes[0])

plt.savefig("fig.svg")
plt.show()

# %%
