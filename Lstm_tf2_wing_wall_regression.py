# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3
# from pathlib import Path
from helper_functions import divide_file_names, data_get_info, data_load, data_shorten_sequence,\
    model_build_tf, model_fit_tf, model_predict_tf, model_evaluate_regression_tf, model_k_fold_tf
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow import keras
# from pandas import DataFrame
# %load_ext autoreload
# %autoreload 2

# %% design parameters
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

lstm_layers = 2
lstm_units = 16  # number of lstm cells of each lstm layer
lr = 0.0001  # learning rate
dropout = 0.0
recurrent_dropout = 0.0
epochs_number = 5000  # number of epochs
epochs_patience = 1000  # for early stopping, set <0 to disable
k_fold_splits = len(sets_train)

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2022.01.31_dropout_recurrent/'  # include trailing slash
save_filename = 'Ro={}_A={}_Tr={}_Te={}_in={}_bl={}_Nc={}_Ns={}_win={}_trun={}_{}L{}_lr={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
    ','.join(str(temp) for temp in inputs_ft), baseline_d, N_cycles_example, N_cycles_step,
    average_window, truncate_sequence, lstm_layers, lstm_units, lr)

# %% get the file names to load data from
file_names, file_labels, file_sets,\
    file_names_train, file_labels_train, file_sets_train,\
    file_names_val, file_labels_val, file_sets_val,\
    baseline_file_names_train, baseline_file_names_val, baseline_file_names = \
    divide_file_names(sets_train, d_train, d_train_labels,
                      sets_val, d_val, d_val_labels,
                      baseline_d,
                      Ro, A_star)

# %% get info about the data
N_files_all, N_files_train, N_files_val,\
    N_examples, N_examples_train, N_examples_val,\
    N_per_example, N_per_step, N_total,\
    N_inputs, N_inputs_ft, N_inputs_ang = \
    data_get_info(data_folder,
                  file_names, file_labels,
                  file_names_train, file_names_val,
                  train_val_split, separate_val_files,
                  N_cycles_example, N_cycles_step, N_cycles_to_use,
                  inputs_ft, inputs_ang)

# %% get training and validation datasets
X_train, y_train, X_val, y_val = \
    data_load(data_folder,
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
X_train, X_val, N_per_example = \
    data_shorten_sequence(X_train, X_val, N_per_example,
                          average_window, truncate_sequence)

# %% initialize the model
# model, callbacks_list = \
#     model_build_tf(lstm_layers, lstm_units, epochs_patience, lr,
#                    dropout, recurrent_dropout,
#                    save_model, model_checkpoint,
#                    save_folder, save_filename,
#                    N_per_example, N_inputs)

# %% train the model
# model, history = \
#     model_fit_tf(model, callbacks_list, epochs_number,
#                  X_train, y_train, X_val, y_val)

# %% train the model using k-fold CV
model, history = \
    model_k_fold_tf(X_train, y_train,
                    lstm_layers, lstm_units, lr, epochs_number, epochs_patience,
                    dropout, recurrent_dropout,
                    k_fold_splits, shuffle_seed,
                    save_model, model_checkpoint, save_results,
                    save_folder, save_filename,
                    N_per_example, N_inputs, file_labels)

# %% predict on training and testing data using trained model
yhat_train, yhat_val = \
    model_predict_tf(model,
                     save_model, model_checkpoint,
                     save_folder, save_filename,
                     X_train, X_val)

# %% evaluate performance
df, loss_val_total = \
    model_evaluate_regression_tf(history,
                                 y_train, y_val, yhat_train, yhat_val,
                                 save_results, save_folder, save_filename,
                                 file_labels)

# %%
