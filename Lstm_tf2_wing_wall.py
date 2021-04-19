# -*- coding: utf-8 -*-
# %%
from operator import sub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix

# %% design parameters
file_names = ['0', '3']
file_names_offset = 3  # difference in between actual distance and file names
trajectory_name = '30deg'  # choose trajectory name for which to process data

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples

N_inputs = 7  # ft_meas + other inputs

empirical_prediction = True
empirical_prediction_name = '22'
subract_prediction = True

shuffle_examples = False

lr = 0.004  # learning rate
epochs_number = 1000  # number of epochs

save_results = True
folder_name = 'plots/2021.04.16'

# %%
# all files to extract the data from (collected at multiple locations)
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])
file_names_float += file_names_offset  # offset between ruler reading and distance from wing tip to wall

# %%
# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(file_names[0] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(file_names[0] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

N = len(t)  # number of data points

# find points where a new stroke cycle is started
t_s = round(t[1] - t[0], 3)  # sample time
freq = cpg_param[-1, 0]  # store frequencies of each param set
t_cycle = 1 / freq  # stroke cycle time

# calculate number of cycles
t_total = t[-1]  # period of time over which data has been collected for each param set
t_total += t_s  # including first point
t_total = np.around(t_total, decimals=3)

# calculate number of data points per cycle
N_per_cycle = round(t_cycle / t_s)
print('Data points in a cycle:', N_per_cycle)

# N_cycles = 50
N_cycles = N // N_per_cycle  # floor(total data points / data points in a cycle)
print('Stroke cycles:', N_cycles)
print('Unused data points:', N - N_per_cycle * N_cycles)  # print number of unused data points

N_examples = (N_cycles - N_cycles_example) // N_cycles_step + 1  # int division
print('Total examples:', N_examples)

# number of training and testing stroke cycles
N_examples_train = round(0.75 * N_examples)
N_examples_test = N_examples - N_examples_train
print('Training examples:', N_examples_train)
print('Testing examples:', N_examples_test)

print('Inputs:', N_inputs)

# %%
data = np.zeros((N_files * N_examples * N_cycles_example * N_per_cycle, N_inputs))  # all data, with groups of stroke cycles
x = np.zeros((N_files * N_examples_train, N_inputs, N_cycles_example * N_per_cycle))
y = np.zeros((N_files * N_examples_train))
x_val = np.zeros((N_files * N_examples_test, N_inputs, N_cycles_example * N_per_cycle))
y_val = np.zeros((N_files * N_examples_test))

if empirical_prediction:  # furthest distance from wall as forward model
    ft_pred = np.loadtxt(empirical_prediction_name + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    if not(empirical_prediction):  # use QS model if empirical prediction is not used
        ft_pred = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    if subract_prediction:  # subtract pred from meas?
        ft_meas -= ft_pred

    for i in range(N_examples):
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), 0:6] = \
            ft_meas[:, (i*N_cycles_step * N_per_cycle):((i*N_cycles_step + N_cycles_example) * N_per_cycle)].T  # measured FT
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), 6] = \
            ang_meas[0, (i*N_cycles_step * N_per_cycle):((i*N_cycles_step + N_cycles_example) * N_per_cycle)].T  # stroke angle

# %%
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # normalize

data = data.reshape(N_files * N_examples, N_cycles_example * N_per_cycle, N_inputs)
data = data.transpose(0, 2, 1)  # example -> FT components -> all data points of that example

if shuffle_examples:
    np.random.shuffle(data)  # randomize order of data to be split into train and test sets

# split data into training and testing sets
for k in range(N_files):
    x[k*N_examples_train:(k+1)*N_examples_train, :, :] = data[k*N_examples:(k+1)*N_examples - N_examples_test, :, :]
    y[k*N_examples_train:(k+1)*N_examples_train] = k  # class = file
    x_val[k*N_examples_test:(k+1)*N_examples_test, :, :] = data[k*N_examples + N_examples_train:(k+1)*N_examples, :, :]
    y_val[k*N_examples_test:(k+1)*N_examples_test] = k  # class = file

# %%
model = keras.models.Sequential(
    [
        keras.layers.RNN(keras.layers.LSTMCell(128), return_sequences=True, input_shape=(N_inputs, N_cycles_example * N_per_cycle)),
        keras.layers.RNN(keras.layers.LSTMCell(128)),
        keras.layers.Dense(N_files, activation='softmax')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

# model.summary()

keras.backend.set_value(model.optimizer.learning_rate, lr)
# print("Learning rate:", model.optimizer.learning_rate.numpy())

early_stopping_monitor = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=100,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

history = model.fit(
    x, y,
    validation_data=(x_val, y_val),
    epochs=epochs_number,
    verbose=0,
    callbacks=[early_stopping_monitor]
)

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

cm_train = confusion_matrix(y, np.argmax(model.predict(x), axis=-1))
cm_test = confusion_matrix(y_val, np.argmax(model.predict(x_val), axis=-1))

if save_results:
    plt.savefig(folder_name + '/lstm_' + str(file_names) + '_(' + str(N_cycles_example) + ',' + str(N_cycles_step) + ')_' + str(lr) + '.png')
    np.savetxt(folder_name + '/lstm_' + str(file_names) + '_(' + str(N_cycles_example) + ',' + str(N_cycles_step) + ')_' + str(lr) + '_train.txt', cm_train, fmt='%d')
    np.savetxt(folder_name + '/lstm_' + str(file_names) + '_(' + str(N_cycles_example) + ',' + str(N_cycles_step) + ')_' + str(lr) + '_test.txt', cm_test, fmt='%d')

print(cm_train)
print(cm_test)
# print(model.predict(x_val))

plt.show()

# %%
