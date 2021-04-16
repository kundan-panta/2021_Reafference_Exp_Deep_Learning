# -*- coding: utf-8 -*-
# %%
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix

# %%
# all files to extract the data from (collected at multiple locations)
file_names = ['0', '6', '12', '18']
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])
file_names_float += 3  # offset between ruler reading and distance from wing tip to wall

# choose trajectory name for which to process data
trajectory_name = '30deg'

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
N_cycles = floor(N / N_per_cycle)  # floor(total data points / data points in a cycle)
print('Stroke cycles:', N_cycles)
print('Unused data points:', N - N_per_cycle * N_cycles)  # print number of unused data points

# group cycles together
N_cycles_example = 10  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
N_examples = (N_cycles - N_cycles_example) // N_cycles_step + 1  # int division
print('Total examples:', N_examples)

# number of training and testing stroke cycles
N_examples_train = round(0.75 * N_examples)
N_examples_test = N_examples - N_examples_train
print('Training examples:', N_examples_train)
print('Testing examples:', N_examples_test)

N_inputs = 7
print('Inputs:', N_inputs)

# %%
data = np.zeros((N_files * N_examples * N_cycles_example * N_per_cycle, N_inputs))  # all data, with groups of stroke cycles
x = np.zeros((N_files * N_examples_train, N_inputs, N_cycles_example * N_per_cycle))
y = np.zeros((N_files * N_examples_train))
x_val = np.zeros((N_files * N_examples_test, N_inputs, N_cycles_example * N_per_cycle))
y_val = np.zeros((N_files * N_examples_test))

############### alternative nominal FT - futhest from wall ###############
ft_pred = np.loadtxt('22/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    # ft_pred = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    ############### take difference?? ###############
    # ft_meas -= ft_pred

    for i in range(N_examples):
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), 0:6] = \
            ft_meas[:, (i*N_cycles_step * N_per_cycle):((i*N_cycles_step + N_cycles_example) * N_per_cycle)].T  # measured FT
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), 6] = \
            ang_meas[0, (i*N_cycles_step * N_per_cycle):((i*N_cycles_step + N_cycles_example) * N_per_cycle)].T  # stroke angle

# %%
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # normalize

data = data.reshape(N_files * N_examples, N_cycles_example * N_per_cycle, N_inputs)
data = data.transpose(0, 2, 1)  # example -> FT components -> all data points of that example

# np.random.shuffle(data)  # randomize order of data to be split into train and test sets

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
        keras.layers.Dense(N_files, activation='sigmoid')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

# model.summary()

lr = 0.0002
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
    epochs=1000,
    verbose=0,
    callbacks=[early_stopping_monitor]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('plots/2021.04.16/lstm_' + str(file_names) + '_(' + str(N_cycles_example) + ',' + str(N_cycles_step) + ')_' + str(lr) + '.png')  # change this
plt.show()

# %%
print(confusion_matrix(y, np.argmax(model.predict(x), axis=-1)))
print(confusion_matrix(y_val, np.argmax(model.predict(x_val), axis=-1)))

# %%
