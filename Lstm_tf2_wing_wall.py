# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.math import confusion_matrix

# %% design parameters
root_folder = ''  # include trailing slash
file_names = ['0', '6', '12', '18']
file_names_offset = 3  # difference in between actual distance and file names
trajectory_name = '30deg'  # choose trajectory name for which to process data

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
N_inputs = 7  # ft_meas + other inputs

empirical_prediction = True  # whether to use collected data as the "perfect prediction"
empirical_prediction_name = '22'
subract_prediction = False  # meas - pred?

shuffle_examples = False

cells_number = 128  # number of lstm cells of each lstm layer
lr = 0.0005  # learning rate
epochs_number = 400  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped

save_plot = True
save_cm = True  # save confusion matrix
save_folder = 'plots/2021.04.22_multi-class/'  # include trailing slash
save_filename = root_folder + save_folder + 'lstm_' + str(file_names) + '_(' + str(N_cycles_example) + ',' + str(N_cycles_step) + ')_3layer' + str(cells_number) + '_' + str(lr) + '_uf'

# %%
# all files to extract the data from (collected at multiple locations)
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])
file_names_float += file_names_offset  # offset between ruler reading and distance from wing tip to wall

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(root_folder + file_names[0] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(root_folder + file_names[0] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

N_total = len(t)  # number of data points

# find points where a new stroke cycle is started
t_s = round(t[1] - t[0], 3)  # sample time
freq = cpg_param[-1, 0]  # store frequency of param set
t_cycle = 1 / freq  # stroke cycle time

t_total = t[-1]  # period of time over which data has been collected for each param set
t_total += t_s  # including first point
t_total = np.around(t_total, decimals=3)

N_per_cycle = round(t_cycle / t_s)  # calculate number of data points per cycle, round instead of floor

# N_cycles = 50
N_cycles = N_total // N_per_cycle  # floor(total data points / data points in a cycle)

N_examples = (N_cycles - N_cycles_example) // N_cycles_step + 1  # int division

# number of training and testing stroke cycles
N_examples_train = round(0.8 * N_examples)
N_examples_test = N_examples - N_examples_train

print('Data points in a cycle:', N_per_cycle)
print('Unused data points:', N_total - N_per_cycle * N_cycles)  # print number of unused data points
print('Stroke cycles:', N_cycles)
print('Total examples:', N_examples)
print('Training examples:', N_examples_train)
print('Testing examples:', N_examples_test)
print('Inputs:', N_inputs)

# %%
data = np.zeros((N_files * N_examples * N_cycles_example * N_per_cycle, N_inputs))  # all data
x = np.zeros((N_files * N_examples_train, N_inputs, N_cycles_example * N_per_cycle))
y = np.zeros((N_files * N_examples_train))
x_val = np.zeros((N_files * N_examples_test, N_inputs, N_cycles_example * N_per_cycle))
y_val = np.zeros((N_files * N_examples_test))

if empirical_prediction:  # furthest distance from wall as forward model
    ft_pred = np.loadtxt(root_folder + empirical_prediction_name + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(root_folder + file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    if not(empirical_prediction):  # use QS model if empirical prediction is not used
        ft_pred = np.loadtxt(root_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(root_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(root_folder + file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(root_folder + file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    if subract_prediction:  # subtract pred from meas?
        ft_meas -= ft_pred

    for i in range(N_examples):
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), 0:6] = \
            ft_meas[:, (i*N_cycles_step * N_per_cycle):((i*N_cycles_step + N_cycles_example) * N_per_cycle)].T  # measured FT
        data[((k*N_examples + i) * N_cycles_example * N_per_cycle):((k*N_examples + i + 1) * N_cycles_example * N_per_cycle), -1] = \
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
        keras.layers.RNN(keras.layers.LSTMCell(cells_number), return_sequences=True, input_shape=(N_inputs, N_cycles_example * N_per_cycle)),
        # keras.layers.RNN(keras.layers.LSTMCell(cells_number), return_sequences=True),
        keras.layers.RNN(keras.layers.LSTMCell(cells_number)),
        # keras.layers.Dense(cells_number),
        keras.layers.Dense(N_files, activation='softmax')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

# model.summary()

keras.backend.set_value(model.optimizer.learning_rate, lr)
print("Learning rate:", model.optimizer.learning_rate.numpy())

# early_stopping_monitor = EarlyStopping(
#     monitor='val_accuracy',
#     mode='auto',
#     min_delta=0,
#     patience=epochs_patience,
#     baseline=None,
#     restore_best_weights=True,
#     verbose=0
# )

model_checkpoint_monitor = ModelCheckpoint(
    save_filename + '.h5',
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True,
    verbose=0
)

history = model.fit(
    x, y,
    validation_data=(x_val, y_val),
    epochs=epochs_number,
    verbose=0,
    callbacks=[model_checkpoint_monitor],
    shuffle=True,
    workers=1,
    use_multiprocessing=False
)

model = keras.models.load_model(save_filename + '.h5')  # load best weights

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

cm_train = confusion_matrix(y, np.argmax(model.predict(x), axis=-1))
cm_test = confusion_matrix(y_val, np.argmax(model.predict(x_val), axis=-1))

if save_plot:
    plt.savefig(save_filename + '.png')
if save_cm:
    np.savetxt(save_filename + '_train.txt', cm_train, fmt='%d')
    np.savetxt(save_filename + '_test.txt', cm_test, fmt='%d')

print(cm_train)
print(cm_test)
# print(model.predict(x_val))

plt.show()

# %%
