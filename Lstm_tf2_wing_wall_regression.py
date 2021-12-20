# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3
# from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from pandas import DataFrame
from helper_functions import divide_file_names, data_get_info, data_load, data_process,\
    model_build_tf, model_fit_tf, model_predict_tf, model_evaluate_regression

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/butterworth_h0.04_l5_o10/'  # include trailing slash
Ro = 3.5
A_star = 2

sets_train = [1, 2, 4, 5, 101]
d_train = [list(range(1, 43 + 1, 3))] * 4 + [list(range(1, 37 + 1, 3))] * 1  # list of all distances from wall for each set
d_train_labels = d_train

sets_val = [3]
d_val = [list(range(1, 43 + 1, 3))]  # list of all distances from wall
d_val_labels = d_val

separate_val_files = len(sets_val) > 0
if separate_val_files:
    train_val_split = 1
    shuffle_examples = False
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
inputs_ang = [0]

baseline_d = None  # set to None for no baseline

lstm_units = 64  # number of lstm cells of each lstm layer
lr = 0.0001  # learning rate
epochs_number = 1500  # number of epochs
epochs_patience = -1  # for early stopping, set <0 to disable

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2021.10.11_new plot code/'  # include trailing slash
# save_filename = ','.join(file_names_train) + '_' + ','.join(file_names_val) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2l' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# save_filename = 'all_' + ','.join(str(temp) for temp in file_labels_val) + '_' + ','.join(file_names_val) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2g' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
save_filename = 'Ro={}_A={}_Tr={}_Te={}_in={}_bl={}_Nc={}_Ns={}_2L{}_lr={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
    ','.join(str(temp) for temp in inputs_ft), baseline_d, N_cycles_example, N_cycles_step, lstm_units, lr)

# %%
file_names, file_labels,\
    file_names_train, file_labels_train,\
    file_names_val, file_labels_val,\
    baseline_file_names_train, baseline_file_names_val,\
    baseline_file_names = divide_file_names(sets_train, d_train, d_train_labels,
                                            sets_val, d_val, d_val_labels,
                                            baseline_d,
                                            Ro, A_star)

# %%
N_files_all, N_files_train, N_files_val,\
    N_examples, N_examples_train, N_examples_val,\
    N_per_example, N_per_step, N_total,\
    N_inputs, N_inputs_ft, N_inputs_ang = data_get_info(data_folder,
                                                        file_names, file_labels,
                                                        file_names_train, file_names_val,
                                                        train_val_split, separate_val_files,
                                                        N_cycles_example, N_cycles_step, N_cycles_to_use,
                                                        inputs_ft, inputs_ang)

# %%
data, labels = data_load(data_folder,
                         file_names, file_labels,
                         baseline_d, baseline_file_names,
                         inputs_ft, inputs_ang,
                         N_files_all, N_examples,
                         N_per_example, N_per_step,
                         N_inputs, N_inputs_ft, N_inputs_ang)

# %%
X_train, y_train, X_val, y_val = data_process(data, labels,
                                              separate_val_files, shuffle_examples, shuffle_seed,
                                              save_model, save_results, save_folder, save_filename,
                                              N_files_all, N_files_train, N_examples,
                                              N_examples_train, N_examples_val, N_per_example, N_inputs)

# %%
model, callbacks_list = model_build_tf(lstm_units, epochs_patience, lr,
                                       save_model, model_checkpoint,
                                       save_folder, save_filename,
                                       N_per_example, N_inputs)

# %%
history = model_fit_tf(model, callbacks_list, epochs_number,
                       X_train, y_train, X_val, y_val)

# %% predict distance to wall
yhat_train, yhat_val = model_predict_tf(model,
                                        save_model, model_checkpoint,
                                        save_folder, save_filename,
                                        X_train, X_val)

# %% evaluate performance
df = model_evaluate_regression(history,
                               y_train, y_val, yhat_train, yhat_val,
                               save_results, save_folder, save_filename,
                               file_labels)

# %%
