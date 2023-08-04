# %%
# python == 3.8.7
# tensorflow == 2.8.0
# numpy == 1.19.3

# %%
from helper_functions import divide_file_names, data_get_info, data_load, data_process, y_norm_reverse, data_full_process
import matplotlib.pyplot as plt
import numpy as np
from os.path import isdir
from tensorflow import keras
# from pandas import DataFrame
from pathlib import Path
import shap
import tensorflow as tf
import pickle
tf.compat.v1.disable_v2_behavior()
# %load_ext autoreload
# %autoreload 2

# %% for changing gpu memory limit
# method 2
# gpus = tf.config.list_physical_devices('GPU')
# print('GPUs:', gpus)
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)

#     except RuntimeError as e:
#         print(e)

# method 3
# gpus = tf.config.list_physical_devices('GPU')
# print('GPUs:', gpus)
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=int(3.5 * 1024))])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# %% turn into function


def shap_apply(data_folder, save_folder, parameters):
    # %% design parameters
    Ro, A_star, sets_val, sets_test, average_window, lstm_layers, dense_hidden_layers, N_units, lr, dropout, shuffle_seed = parameters

    # root_folder = ''  # include trailing slash
    # data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
    # Ro = 3.5
    # A_star = 2
    Ro_d_last = {2: 40, 3.5: 40, 5: 40}  # furthest distance from wall for each wing shape

    # all sets except the ones given in sets_test
    sets_train = [1, 2, 3, 4, 5]
    [sets_train.remove(set_val) for set_val in sets_val if set_val in sets_train]
    [sets_train.remove(set_test) for set_test in sets_test if set_test in sets_train]

    # sets_train = [1, 2, 3, 4]
    d_train = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_train)  # list of all distances from wall for each set
    d_train_labels = d_train

    # sets_val = [3]
    d_val = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_val)  # list of all distances from wall
    d_val_labels = d_val

    # sets_test = [5]
    d_test = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_test)  # list of all distances from wall
    d_test_labels = d_test

    separate_val_files = len(sets_val) > 0
    if separate_val_files:
        train_val_split = 1
        # shuffle_examples = False
        # shuffle_seed = None
    else:
        train_val_split = 0.8
        # shuffle_examples = True
        # shuffle_seed = np.random.default_rng().integers(0, high=1000)
        # shuffle_seed = 5  # seed to split data in reproducible way

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
    # average_window = 10
    baseline_d = None  # set to None for no baseline

    # lstm_layers = 2
    # dense_hidden_layers = 1
    # N_units = 16  # number of lstm cells of each lstm layer
    # lr = 0.0002  # learning rate
    # dropout = 0.2
    recurrent_dropout = 0.0
    epochs_number = 5000  # number of epochs
    epochs_patience = 10000  # for early stopping, set <0 to disable
    # k_fold_splits = len(sets_train)

    save_model = False  # save model file, save last model if model_checkpoint == False
    model_checkpoint = False  # doesn't do anything if save_model == False
    save_results = False
    # save_folder = root_folder + save_folder  # include trailing slash
    save_filename = 'Ro={}_A={}_Tr={}_Val={}_Te={}_inF={}_inA={}_bl={}_Ne={}_Ns={}_win={}_sh={}_{}L{}D{}_lr={}_dr={}'.format(
        Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
        ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft), ','.join(str(temp) for temp in inputs_ang),
        baseline_d, N_cycles_example, N_cycles_step, average_window, shuffle_seed,
        lstm_layers, dense_hidden_layers, N_units, lr, dropout, recurrent_dropout)

    # %% get the saved pre-processing info
    if not(isdir(save_folder + save_filename)):
        return

    model = keras.models.load_model(save_folder + save_filename)
    # model.summary()

    # %% initialize
    if norm_X:
        X_mean = np.loadtxt(data_folder + 'Ro={}/A={}/X_mean.txt'.format(Ro, A_star))
        X_std = np.loadtxt(data_folder + 'Ro={}/A={}/X_std.txt'.format(Ro, A_star))
    else:
        X_mean, X_std = [None, None]

    if norm_y:
        y_mean = np.loadtxt(data_folder + 'Ro={}/A={}/y_mean.txt'.format(Ro, A_star))
        y_std = np.loadtxt(data_folder + 'Ro={}/A={}/y_std.txt'.format(Ro, A_star))
    else:
        y_mean, y_std = [None, None]

    if baseline_d is None:
        X_baseline = None
    else:
        X_baseline = np.loadtxt(save_folder + save_filename + '/X_baseline.txt')

    # if len(sets_val) == 0 or len(sets_test) == 0:
    #     shuffle_seed = int(np.loadtxt(save_folder + save_filename + '/shuffle_seed.txt'))

    # %% load data
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

    # %% get shap values
    # select a set of background examples to take an expectation over
    # background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    background = X_train

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)

    shap_values = e.shap_values(X_test)

    # %% plot shap values
    shap_val = shap_values
    shap_val = np.array(shap_val)
    shap_val = np.reshape(shap_val, (int(shap_val.shape[1]), int(shap_val.shape[2]), int(shap_val.shape[3])))

    # %% save shap
    # save_folder += 'shap/'
    # Path(save_folder + save_filename).mkdir(parents=True, exist_ok=True)  # make folder
    np.save(save_folder + save_filename + "/shap.npy", shap_val)
    shap_file = open(save_folder + save_filename + "/shap", 'wb')
    pickle.dump(shap_val, shap_file)
    shap_file.close()

# %%
