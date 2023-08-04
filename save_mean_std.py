# %%
# from helper_functions import divide_file_names, data_get_info, data_load, data_shorten_sequence
from helper_functions import data_full_process, y_norm_reverse
# from helper_functions import model_k_fold_tf
# from helper_functions import model_build_tf, model_fit_tf
# from helper_functions import model_predict_tf, model_evaluate_regression_tf
# import matplotlib.pyplot as plt
import numpy as np
# %load_ext autoreload
# %autoreload 2

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
Ro = 2
A_star = 4
Ro_d_last = {2: 40, 3.5: 40, 5: 40}  # furthest distance from wall for each wing shape

# all sets except the ones given in sets_val
# sets_train = [1, 2, 3, 4, 5]
# [sets_train.remove(set_val) for set_val in sets_val if set_val in sets_train]
# [sets_train.remove(set_test) for set_test in sets_test if set_test in sets_train]

sets_train = [1, 2, 3, 4, 5]
d_train = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_train)  # list of all distances from wall for each set
d_train_labels = d_train

sets_val = []
d_val = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_val)  # list of all distances from wall
d_val_labels = d_val

sets_test = []
d_test = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_test)  # list of all distances from wall
d_test_labels = d_test

separate_val_files = len(sets_val) > 0
if separate_val_files:
    train_val_split = 1
    # shuffle_examples = False
    shuffle_seed = None
else:
    train_val_split = 0.8
    # shuffle_examples = True
    # shuffle_seed = np.random.default_rng().integers(0, high=1000)
    shuffle_seed = 5  # seed to split data in reproducible way

separate_test_files = len(sets_test) > 0
if separate_test_files:
    train_test_split = 1
    # shuffle_examples = False
    # shuffle_seed = None
else:
    train_test_split = 0.8
    # shuffle_examples = True
    # shuffle_seed = np.random.default_rng().integers(0, high=1000)
    # shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 14

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]

norm_X = True
norm_y = True
average_window = 10
baseline_d = None  # set to None for no baseline

lstm_layers = 2
dense_hidden_layers = 1
N_units = 16  # number of lstm cells of each lstm layer
lr = 0.0002  # learning rate
dropout = 0.2
recurrent_dropout = 0.0
epochs_number = 5000  # number of epochs
epochs_patience = 10000  # for early stopping, set <0 to disable
# k_fold_splits = len(sets_train)

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2022.03.09_data_plot_new/'  # include trailing slash
save_filename = 'Ro={}_A={}_Tr={}_Val={}_Te={}_in={}_bl={}_Ne={}_Ns={}_win={}_{}L{}D{}_lr={}_dr={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
    ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft),
    baseline_d, N_cycles_example, N_cycles_step, average_window,
    lstm_layers, dense_hidden_layers, N_units, lr, dropout, recurrent_dropout)

# %% load the data
train_val_split = 1
train_test_split = 1
[X_mean, X_std, y_mean, y_std, X_baseline] = [None, None, None, None, None]  # initialize

X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,\
    X_mean, X_std, y_mean, y_std, X_baseline, N_per_example, N_inputs, t_s, t_cycle = \
    data_full_process(
        data_folder, Ro, A_star,
        sets_train, d_train, d_train_labels,
        sets_val, d_val, d_val_labels,
        sets_test, d_test, d_test_labels,
        inputs_ft, inputs_ang,
        N_cycles_example, N_cycles_step, N_cycles_to_use,
        separate_val_files, train_val_split, shuffle_seed,
        separate_test_files, train_test_split,
        save_model, save_folder, save_filename,
        norm_X, norm_y, X_mean, X_std, y_mean, y_std,
        baseline_d, X_baseline, average_window
    )

# %% save the mean and std data
np.savetxt(data_folder + 'Ro={}/A={}/X_mean.txt'.format(Ro, A_star), np.squeeze(X_mean))
np.savetxt(data_folder + 'Ro={}/A={}/X_std.txt'.format(Ro, A_star), np.squeeze(X_std))
np.savetxt(data_folder + 'Ro={}/A={}/y_mean.txt'.format(Ro, A_star), [y_mean])
np.savetxt(data_folder + 'Ro={}/A={}/y_std.txt'.format(Ro, A_star), [y_std])

# %%
