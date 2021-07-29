# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.math import confusion_matrix

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.07.28/f_a6_s15_o60/'  # include trailing slash

Ro = 5
A_star = 2
d_all = list(range(1, 41, 3))  # list of all distances from wall
d_all_labels = [0] * 9 + [1] * 5
sets_train = [4, 5]
sets_test = [1]

if len(sets_test) > 0:
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 0.8
    shuffle_examples = True

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]

# empirical_prediction = True  # whether to use collected data as the "perfect prediction"
# empirical_prediction_name = '22'
# subract_prediction = False  # meas - pred?

lstm_units = 64  # number of lstm cells of each lstm layer
lr = 0.01  # learning rate
epochs_number = 300  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped

save_plot = True
save_cm = True  # save confusion matrix
save_model = True  # save model file
save_folder = 'plots/2021.07.28_new_data_gru/'  # include trailing slash
# save_filename = root_folder + save_folder + ','.join(file_names_train) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2l' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# save_filename = root_folder + save_folder + 'all_' + ','.join(str(temp) for temp in file_labels_test) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2g' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
save_filename = root_folder + save_folder + 'Ro={:s}_A={:s}_d={:s}_Tr={:s}_Te={:s}_in={:s}_Nc={:s}_Ns={:s}_2g{:d}_lr={:s}'.format(str(Ro), str(A_star), ','.join(str(temp) for temp in d_all_labels), ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft), str(N_cycles_example), str(N_cycles_step), lstm_units, str(lr))

# %%
# get the file names and labels
file_names_train = []
file_labels_train = []
for s in sets_train:
    for d_index, d in enumerate(d_all):
        file_names_train.append('Ro={:s}/A={:s}/Set={:d}/d={:d}/'.format(str(Ro), str(A_star), s, d))
        file_labels_train.append(d_all_labels[d_index])

file_names_test = []
file_labels_test = []
for s in sets_test:
    for d_index, d in enumerate(d_all):
        file_names_test.append('Ro={:s}/A={:s}/Set={:d}/d={:d}/'.format(str(Ro), str(A_star), s, d))
        file_labels_test.append(d_all_labels[d_index])

file_names = file_names_train + file_names_test
file_labels = file_labels_train + file_labels_test
#

N_files_train = len(file_names_train)
N_files_test = len(file_names_test)
if not(len(sets_test) > 0):
    N_files_test = N_files_train
N_files_all = len(file_names)

assert len(file_labels) == N_files_all  # makes sure labels are there for all files

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(root_folder + data_folder + file_names[0] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(root_folder + data_folder + file_names[0] + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

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
N_examples_train = round(train_test_split * N_examples)
if len(sets_test) > 0:
    N_examples_test = N_examples
else:
    N_examples_test = N_examples - N_examples_train

N_inputs_ft = len(inputs_ft)
N_inputs_ang = len(inputs_ang)
N_inputs = N_inputs_ft + N_inputs_ang  # ft_meas + other inputs

N_classes = len(np.unique(file_labels))

print('Frequency:', freq)
print('Data points in an example:', N_per_example)
print('Unused data points:', N_total - ((N_examples - 1) * N_per_step + N_per_example))  # print number of unused data points
print('Total examples per file:', N_examples)
print('Training examples per file:', N_examples_train)
print('Testing examples per file:', N_examples_test)
print('Inputs:', N_inputs)
print('Clases:', N_classes)

# %%
data = np.zeros((N_files_all * N_examples * N_per_example, N_inputs))  # all input data
labels = np.zeros((N_files_all * N_examples), dtype=int)  # all labels

# if empirical_prediction:  # furthest distance from wall as forward model
#     ft_pred = np.loadtxt(root_folder + data_folder + empirical_prediction_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files_all):
    # get data
    t = np.around(np.loadtxt(root_folder + data_folder + file_names[k] + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    # if not(empirical_prediction):  # use QS model if empirical prediction is not used
    #     ft_pred = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

    # if subract_prediction:  # subtract pred from meas?
    #     ft_meas -= ft_pred

    for i in range(N_examples):
        data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), :N_inputs_ft] = \
            ft_meas[inputs_ft, (i * N_per_step):(i * N_per_step + N_per_example)].T  # measured FT
        if N_inputs_ang > 0:
            data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), N_inputs_ft:] = \
                ang_meas[inputs_ang, (i * N_per_step):(i * N_per_step + N_per_example)].T  # stroke angle
        labels[k * N_examples + i] = file_labels[k]
        # sanity checks for data: looked at 1st row of 1st file, last row of 1st file, first row of 2nd file,
        # last row of last file, to make sure all the data I needed was at the right place

# %%
# save the min and max values used for normalization of the data
Path(save_filename).mkdir(parents=True, exist_ok=True)  # make folder
np.savetxt(save_filename + '/data_min.txt', np.min(data, axis=0))
np.savetxt(save_filename + '/data_max.txt', np.max(data, axis=0))

data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # normalize
data = data.reshape(N_files_all * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
# data = data.transpose(0, 2, 1)  # feature major

if shuffle_examples:  # randomize order of data to be split into train and test sets
    permutation = list(np.random.permutation(N_files_all * N_examples))
    data = data[permutation]
    labels = labels[permutation]

# split data into training and testing sets
X_train = data[:N_files_train * N_examples_train]
y_train = labels[:N_files_train * N_examples_train]
X_test = data[N_files_train * N_examples_train:]
y_test = labels[N_files_train * N_examples_train:]

# %%
model = keras.models.Sequential(
    [
        # keras.layers.Conv1D(conv_filters, conv_kernel_size, activation='relu', input_shape=(N_per_example, N_inputs)),
        # keras.layers.Conv1D(N_inputs, 3, activation='relu'),
        # keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs)),
        # keras.layers.LSTM(lstm_units),
        keras.layers.GRU(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs)),
        keras.layers.GRU(lstm_units),
        # keras.layers.RNN(keras.layers.LSTMCell(lstm_units), return_sequences=True, input_shape=(N_per_example, N_inputs)),
        # keras.layers.RNN(keras.layers.LSTMCell(lstm_units)),
        # keras.layers.SimpleRNN(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs), unroll=True),
        # keras.layers.SimpleRNN(lstm_units),
        keras.layers.Dense(N_classes)  # , activation='softmax')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
    # steps_per_execution=100
)
keras.backend.set_value(model.optimizer.learning_rate, lr)
print("Learning rate:", model.optimizer.learning_rate.numpy())

model.summary()

callbacks_list = []

# early_stopping_monitor = EarlyStopping(
#     monitor='val_accuracy',
#     mode='auto',
#     min_delta=0,
#     patience=epochs_patience,
#     baseline=None,
#     restore_best_weights=True,
#     verbose=0
# )

if save_model:
    model_checkpoint_monitor = ModelCheckpoint(
        save_filename,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True,
        verbose=0
    )
    callbacks_list.append(model_checkpoint_monitor)

# %%
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs_number,
    verbose=0,
    callbacks=callbacks_list,
    shuffle=True,
    workers=1,
    use_multiprocessing=False
)

# %%
plt.rcParams.update({"savefig.facecolor": (1.0, 1.0, 1.0, 1)})  # disable transparent background
plt.tight_layout()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

# make and save the confusion matrix twice
cm_train = confusion_matrix(y_train, np.argmax(model.predict(X_train), axis=-1))
cm_test = confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=-1))
if save_cm:
    np.savetxt(save_filename + '/cm_train_last.txt', cm_train, fmt='%d')
    np.savetxt(save_filename + '/cm_test_last.txt', cm_test, fmt='%d')
print('Last:')
print('Train accuracy: {:.1f}%\tTest accuracy: {:.1f}%'.format(np.trace(cm_train) / np.sum(cm_train) * 100, np.trace(cm_test) / np.sum(cm_test) * 100))
print(cm_train)
print(cm_test)
print("Predictions (Test):")
print(np.argmax(model.predict(X_test), axis=-1))

if save_model:  # load best weights for test accuracy
    model = keras.models.load_model(save_filename)
    # confusion matrix again for best test weights
    cm_train = confusion_matrix(y_train, np.argmax(model.predict(X_train), axis=-1))
    cm_test = confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=-1))
    if save_cm:
        np.savetxt(save_filename + '/cm_train_best.txt', cm_train, fmt='%d')
        np.savetxt(save_filename + '/cm_test_best.txt', cm_test, fmt='%d')
    print('Best:')
    print('Train accuracy: {:.1f}%\tTest accuracy: {:.1f}%'.format(np.trace(cm_train) / np.sum(cm_train) * 100, np.trace(cm_test) / np.sum(cm_test) * 100))
    print(cm_train)
    print(cm_test)
    print("Predictions (Test):")
    print(np.argmax(model.predict(X_test), axis=-1))

if save_plot:
    plt.savefig(save_filename + '.png')
plt.show()

# %%
