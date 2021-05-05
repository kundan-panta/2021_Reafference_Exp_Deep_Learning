# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.math import confusion_matrix

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.05.03/filtered_a10_s22_o20/'  # include trailing slash
file_names = ['0', '6', '12', '18', '24']
file_names_offset = 2  # difference in between actual distance and file names
trajectory_name = '30deg'  # choose trajectory name for which to process data

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = N_cycles_example  # number of cycles to step between consecutive examples

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]

# empirical_prediction = True  # whether to use collected data as the "perfect prediction"
# empirical_prediction_name = '22'
# subract_prediction = False  # meas - pred?

train_test_split = 0.8
shuffle_examples = True

cells_number = 128  # number of lstm cells of each lstm layer
lr = 0.0001  # learning rate
epochs_number = 500  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped

save_plot = True
save_cm = True  # save confusion matrix
save_model = True  # save model file
save_folder = 'plots/2021.05.03_filter/'  # include trailing slash
save_filename = root_folder + save_folder + ','.join(file_names) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_' + str(lr) + '_f10,22,20'

# %%
# all files to extract the data from (collected at multiple locations)
N_files = len(file_names)

# also convert the list into an array of floats
# file_names_float = np.zeros(N_files)
# for i in range(N_files):
#     file_names_float[i] = float(file_names[i])
# file_names_float += file_names_offset  # offset between ruler reading and distance from wing tip to wall

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(root_folder + data_folder + file_names[0] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(root_folder + data_folder + file_names[0] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

N_total = len(t)  # number of data points

t_s = round(t[1] - t[0], 3)  # sample time
freq = cpg_param[-1, 0]  # store frequency of param set
t_cycle = 1 / freq  # stroke cycle time

N_per_example = round(N_cycles_example * t_cycle / t_s)  # number of data points per cycle, round instead of floor
N_per_step = round(N_cycles_step * t_cycle / t_s)
N_examples = (N_total - N_per_example) // N_per_step + 1  # floor division
assert N_total >= (N_examples * N_per_example)  # must not exceed total number of data points

# number of training and testing stroke cycles
N_examples_train = round(train_test_split * N_examples)
N_examples_test = N_examples - N_examples_train

N_inputs_ft = len(inputs_ft)
N_inputs_ang = len(inputs_ang)
N_inputs = N_inputs_ft + N_inputs_ang  # ft_meas + other inputs

print('Data points in an example:', N_per_example)
print('Unused data points:', N_total - N_examples * N_per_example)  # print number of unused data points
print('Total examples per file:', N_examples)
print('Training examples per file:', N_examples_train)
print('Testing examples per file:', N_examples_test)
print('Inputs:', N_inputs)

# %%
data = np.zeros((N_files * N_examples * N_per_example, N_inputs))  # all input data
labels = np.zeros((N_files * N_examples), dtype=int)  # all labels

# if empirical_prediction:  # furthest distance from wall as forward model
#     ft_pred = np.loadtxt(root_folder + data_folder + empirical_prediction_name + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    # if not(empirical_prediction):  # use QS model if empirical prediction is not used
    #     ft_pred = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

    # if subract_prediction:  # subtract pred from meas?
    #     ft_meas -= ft_pred

    for i in range(N_examples):
        data[((k*N_examples + i) * N_per_example):((k*N_examples + i + 1) * N_per_example), 0:6] = \
            ft_meas[:, (i*N_per_step):(i*N_per_step + N_per_example)].T  # measured FT
        data[((k*N_examples + i) * N_per_example):((k*N_examples + i + 1) * N_per_example), -1] = \
            ang_meas[0, (i*N_per_step):(i*N_per_step + N_per_example)].T  # stroke angle
        labels[k*N_examples + i] = k
        # sanity checks for data: looked at 1st row of 1st file, last row of 1st file, first row of 2nd file,
        # last row of last file, to make sure all the data I needed was at the right place

# %%
# save the min and max values used for normalization of the data
Path(save_filename).mkdir(parents=True, exist_ok=True)  # make folder
np.savetxt(save_filename + '/data_min.txt', np.min(data, axis=0))
np.savetxt(save_filename + '/data_max.txt', np.max(data, axis=0))

data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # normalize
data = data.reshape(N_files * N_examples, N_per_example, N_inputs)
data = data.transpose(0, 2, 1)  # example -> FT components -> all data points of that example

if shuffle_examples:  # randomize order of data to be split into train and test sets
    permutation = list(np.random.permutation(N_files * N_examples))
    data = data[permutation]
    labels = labels[permutation]

# split data into training and testing sets
x = data[:N_files*N_examples_train]
y = labels[:N_files*N_examples_train]
x_val = data[N_files*N_examples_train:]
y_val = labels[N_files*N_examples_train:]

# %%
model = keras.models.Sequential(
    [
        keras.layers.RNN(keras.layers.LSTMCell(cells_number), return_sequences=True, input_shape=(N_inputs, N_per_example)),
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
    save_filename,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True,
    verbose=0
)

callbacks_list = []
if save_model:
    callbacks_list.append(model_checkpoint_monitor)

history = model.fit(
    x, y,
    validation_data=(x_val, y_val),
    epochs=epochs_number,
    verbose=0,
    callbacks=callbacks_list,
    shuffle=True,
    workers=1,
    use_multiprocessing=False
)

if save_model:
    model = keras.models.load_model(save_filename)  # load best weights

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
    np.savetxt(save_filename + '/cm_train.txt', cm_train, fmt='%d')
    np.savetxt(save_filename + '/cm_test.txt', cm_test, fmt='%d')

print(cm_train)
print(cm_test)
# print(model.predict(x_val))

plt.show()

# %%
