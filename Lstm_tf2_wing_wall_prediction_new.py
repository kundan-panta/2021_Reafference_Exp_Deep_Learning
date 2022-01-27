# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3
# from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.math import confusion_matrix
from pandas import DataFrame
from helper_functions import divide_file_names, data_get_info, data_load,\
    model_build_tf, model_fit_tf, model_predict_tf, model_evaluate_regression_tf

# %% FOLDERS
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
save_folder = 'plots/2021.12.22_averaged/'  # include trailing slash
save_filename = 'Ro=3.5_A=2_Tr=1,2,4,5_Te=3_in=0,1,2,3,4,5_bl=None_Nc=1_Ns=1_2L16_lr=0.0002_win=10_trun=1'
save_filename = root_folder + save_folder + save_filename

# %% FILES TO PREDICT ON
file_names = ['Ro=3.5/A=2/Set=101/d=1',
              'Ro=3.5/A=2/Set=101/d=4',
              'Ro=3.5/A=2/Set=101/d=7',
              'Ro=3.5/A=2/Set=101/d=10',
              'Ro=3.5/A=2/Set=101/d=13',
              'Ro=3.5/A=2/Set=101/d=16',
              'Ro=3.5/A=2/Set=101/d=19',
              'Ro=3.5/A=2/Set=101/d=22',
              'Ro=3.5/A=2/Set=101/d=25',
              'Ro=3.5/A=2/Set=101/d=28',
              'Ro=3.5/A=2/Set=101/d=31',
              'Ro=3.5/A=2/Set=101/d=34',
              'Ro=3.5/A=2/Set=101/d=37']  # data to predict on
file_labels = list(range(1, 37 + 1, 3))

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = []
average_window = 10
truncate_sequence = 1

baseline_d = None  # set to None for no baseline
if baseline_d is not None:
    baseline_file_names = ['Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37',
                           'Ro=3.5/A=2/Set=101/d=37']
    assert len(baseline_file_names) == len(file_names)

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0

# %% DUMMY VARIABLES
file_names_train = []
file_names_val = []
sets_val = []
train_val_split = 1
separate_val_files = True

# %%
N_files_train = len(file_names_train)
N_files_val = len(file_names_val)
if not(separate_val_files):  # if separate test files are not provided, then we use all the files for both training and testing
    N_files_val = N_files_train
N_files_all = len(file_names)

assert len(file_labels) == N_files_all  # makes sure labels are there for all files

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(data_folder + file_names[0] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(data_folder + file_names[0] + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

t_s = round(t[1] - t[0], 3)  # sample time
freq = cpg_param[-1, 0]  # store frequency of param set
t_cycle = 1 / freq  # stroke cycle time

if N_cycles_to_use == 0:  # if number of cycles per file is not explicitly specified
    N_total = len(t)  # number of data points
else:
    N_per_cycle = round(t_cycle / t_s)  # number of data points per cycle, round instead of floor
    N_total = N_cycles_to_use * N_per_cycle + 100  # limit amount of data to use

N_per_example = round(N_cycles_example * t_cycle / t_s)  # number of data points per example, round instead of floor
N_per_step = round(N_cycles_step * t_cycle / t_s)
N_examples = (N_total - N_per_example) // N_per_step + 1  # floor division
assert N_total >= (N_examples - 1) * N_per_step + N_per_example  # last data point used must not exceed total number of data points

# number of training and testing stroke cycles
N_examples_train = round(train_val_split * N_examples)
if separate_val_files:
    N_examples_val = N_examples
else:
    N_examples_val = N_examples - N_examples_train

N_inputs_ft = len(inputs_ft)
N_inputs_ang = len(inputs_ang)
N_inputs = N_inputs_ft + N_inputs_ang  # ft_meas + other inputs

# N_classes = len(np.unique(file_labels))
# assert np.max(file_labels) == N_classes - 1  # check for missing labels in between

print('Frequency:', freq)
print('Data points in an example:', N_per_example)
print('Unused data points:', N_total - ((N_examples - 1) * N_per_step + N_per_example))  # print number of unused data points
print('Total examples per file:', N_examples)
print('Training examples per file:', N_examples_train)
print('Testing examples per file:', N_examples_val)
print('Inputs:', N_inputs)
# print('Clases:', N_classes)

# %%
data = np.zeros((N_files_all * N_examples * N_per_example, N_inputs))  # all input data
labels = np.zeros((N_files_all * N_examples))  # , dtype=int)  # all labels

for k in range(N_files_all):
    # get data
    t = np.around(np.loadtxt(data_folder + file_names[k] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    ft_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

    if baseline_d is not None:  # subtract pred from meas?
        baseline_ft_meas = np.loadtxt(data_folder + baseline_file_names[k] + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
        ft_meas -= baseline_ft_meas

    for i in range(N_examples):
        data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), :N_inputs_ft] = \
            ft_meas[inputs_ft, (i * N_per_step):(i * N_per_step + N_per_example)].T  # measured FT
        if N_inputs_ang > 0:
            data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), N_inputs_ft:] = \
                ang_meas[inputs_ang, (i * N_per_step):(i * N_per_step + N_per_example)].T  # stroke angle
        labels[k * N_examples + i] = file_labels[k]
        # sanity checks for data: looked at 1st row of 1st file, last row of 1st file, first row of 2nd file,
        # last row of last file, to make sure all the data I needed was at the right place

# %% GET NORMALIZATION VALUES
data_min = np.reshape(np.loadtxt(save_filename + '/data_min.txt'), (1, -1))
data_max = np.reshape(np.loadtxt(save_filename + '/data_max.txt'), (1, -1))

# %%
data = (data - data_min) / (data_max - data_min)  # normalize

print(np.min(data, axis=0))  # check normalization
print(np.max(data, axis=0))  # check normalization

data = data.reshape(N_files_all * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
# data = data.transpose(0, 2, 1)  # feature major

# %% reduce sequence length
N_per_example = N_per_example // average_window  # update sequence length

# cut out last data points so the number of data points is divisible by average_window
data = data[:, 0:N_per_example * average_window, :]

# reshape the time series so
data = data.reshape(data.shape[0], -1, average_window, data.shape[2]).mean(axis=2)

# %% truncate sequence further
N_per_example = round(N_per_example * truncate_sequence)
data = data[:, 0:N_per_example, :]

print('Data points in an example after averaging:', N_per_example)

# %%
model_best = keras.models.load_model(save_filename)  # load best weights

# %%
# model_prediction = np.argmax(model.predict(data), axis=-1)
# cm = confusion_matrix(labels, model_prediction)
# for p, prediction in enumerate(model_prediction):
#     print(prediction, end=' ')
#     if p % N_examples == N_examples - 1:
#         print('\n')
# print(cm)
# print('{:.2f}% accuracy'.format(np.trace(cm) / np.sum(cm) * 100))

# %%
X_val = data
y_val = labels
d_all_labels = file_labels

yhat_val = np.squeeze(model_best.predict(X_val))

# print model predictions
print("Predictions (Test):")
for p, prediction in enumerate(yhat_val):
    print('{:.1f}\t'.format(prediction), end='')
    if p % N_examples_val == N_examples_val - 1:
        print('\t\t')

# calculate result metrics
mu_val = np.zeros_like(d_all_labels, dtype=float)
std_val = np.zeros_like(d_all_labels, dtype=float)

for d_index, d in enumerate(d_all_labels):
    yhat_val_d = yhat_val[y_val == d]
    mu_val[d_index] = np.mean(yhat_val_d)
    std_val[d_index] = np.std(yhat_val_d)

# for printing
df = DataFrame({"d": d_all_labels,
                "mu_val": mu_val,
                "std_val": std_val,
                "ci_down_val": mu_val - 2 * std_val,
                "ci_up_val": mu_val + 2 * std_val})
print(df.round(1).to_string(index=False))

# %%
