# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pandas import DataFrame

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/butterworth_h0.01_l5_o10/'  # include trailing slash
Ro = 3.5
A_star = 2

sets_train = [1, 2, 3, 4, 5, 101]
d_train = [list(range(1, 43 + 1, 3))] * 5 + [list(range(1, 37 + 1, 3))]  # list of all distances from wall for each set
d_train_labels = d_train
assert len(sets_train) == len(d_train)

sets_test = []
d_test = []  # list of all distances from wall
d_test_labels = d_test
assert len(sets_test) == len(d_test)

separate_test_files = len(sets_test) > 0
if separate_test_files:
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 0.8
    shuffle_examples = True
    shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0

inputs_ft = [0]
inputs_ang = [0]

baseline_d = None  # set to None for no baseline

lstm_units = 128  # number of lstm cells of each lstm layer
lr = 0.00005  # learning rate
epochs_number = 1500  # number of epochs
epochs_patience = 300  # for early stopping, set <0 to disable

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2021.08.21_butterworth/'  # include trailing slash
# save_filename = ','.join(file_names_train) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2l' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# save_filename = 'all_' + ','.join(str(temp) for temp in file_labels_test) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2g' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
save_filename = 'Ro={}_A={}_Tr={}_Te={}_in={}_bl={}_Nc={}_Ns={}_2g{}_lr={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_test),
    ','.join(str(temp) for temp in inputs_ft), baseline_d, N_cycles_example, N_cycles_step, lstm_units, lr)

# %%
# get the file names and labels
file_names_train = []
file_labels_train = []
for s_index, s in enumerate(sets_train):
    for d_index, d in enumerate(d_train[s_index]):
        file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
        file_labels_train.append(d_train_labels[s_index][d_index])

file_names_test = []
file_labels_test = []
for s_index, s in enumerate(sets_test):
    for d_index, d in enumerate(d_test[s_index]):
        file_names_test.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
        file_labels_test.append(d_test_labels[s_index][d_index])

file_names = file_names_train + file_names_test
file_labels = file_labels_train + file_labels_test

# baseline file names for each set
if baseline_d is not None:
    baseline_file_names_train = []
    for s_index, s in enumerate(sets_train):
        for d_index, d in enumerate(d_train[s_index]):
            baseline_file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

    baseline_file_names_test = []
    for s_index, s in enumerate(sets_test):
        for d_index, d in enumerate(d_test[s_index]):
            baseline_file_names_test.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

    baseline_file_names = baseline_file_names_train + baseline_file_names_test
    assert len(baseline_file_names) == len(file_names)

# %%
N_files_train = len(file_names_train)
N_files_test = len(file_names_test)
if not(separate_test_files):  # if separate test files are not provided, then we use all the files for both training and testing
    N_files_test = N_files_train
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
N_examples_train = round(train_test_split * N_examples)
if separate_test_files:
    N_examples_test = N_examples
else:
    N_examples_test = N_examples - N_examples_train

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
print('Testing examples per file:', N_examples_test)
print('Inputs:', N_inputs)
# print('Clases:', N_classes)

if save_model or save_results:
    Path(save_folder + save_filename).mkdir(parents=True, exist_ok=True)  # make folder

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

# %%
data_min = np.min(data, axis=0)
data_max = np.max(data, axis=0)

# save the min and max values used for normalization of the data
if save_model:
    np.savetxt(save_folder + save_filename + '/data_min.txt', data_min)
    np.savetxt(save_folder + save_filename + '/data_max.txt', data_max)

data = (data - data_min) / (data_max - data_min)  # normalize
data = data.reshape(N_files_all * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
# data = data.transpose(0, 2, 1)  # feature major

if shuffle_examples:  # randomize order of data to be split into train and test sets
    if not(separate_test_files):
        # shuffle every N_examples examples
        # then pick the first N_examples_train examples and put it to training set
        # and the remaining (N_examples_test) examples into the testing set
        N_examples_train_all = N_files_train * N_examples_train
        permutation = np.zeros(N_files_all * N_examples, dtype=int)
        for k in range(N_files_all):  # each file has N_example examples, and everything is in order
            shuffled = np.array(np.random.RandomState(seed=shuffle_seed + k).permutation(N_examples), dtype=int)
            permutation[k * N_examples_train:(k + 1) * N_examples_train] = k * N_examples + shuffled[:N_examples_train]
            permutation[N_examples_train_all + k * N_examples_test:N_examples_train_all + (k + 1) * N_examples_test] = k * N_examples + shuffled[N_examples_train:]
    else:
        permutation = list(np.random.RandomState(seed=shuffle_seed).permutation(N_files_all * N_examples))
    data = data[permutation]
    labels = labels[permutation]

# labels = np.eye(N_classes)[labels]  # one-hot labels

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
        keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs)),
        keras.layers.LSTM(lstm_units),
        # keras.layers.GRU(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs)),
        # keras.layers.GRU(lstm_units),
        # keras.layers.RNN(keras.layers.LSTMCell(lstm_units), return_sequences=True, input_shape=(N_per_example, N_inputs)),
        # keras.layers.RNN(keras.layers.LSTMCell(lstm_units)),
        # keras.layers.SimpleRNN(lstm_units, return_sequences=True, input_shape=(N_per_example, N_inputs), unroll=True),
        # keras.layers.SimpleRNN(lstm_units),
        keras.layers.Dense(lstm_units, activation='elu'),
        keras.layers.Dense(1)  # , activation='exponential')
    ]
)

model.compile(
    loss=keras.losses.LogCosh(),
    optimizer="adam",
    # metrics=["accuracy"],
    # steps_per_execution=100
)
keras.backend.set_value(model.optimizer.learning_rate, lr)
print("Learning rate:", model.optimizer.learning_rate.numpy())

model.summary()

callbacks_list = []

if epochs_patience > -1:
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        min_delta=0,
        patience=epochs_patience,
        baseline=None,
        restore_best_weights=True,
        verbose=0
    )
    callbacks_list.append(early_stopping_monitor)

if save_model and model_checkpoint:
    model_checkpoint_monitor = ModelCheckpoint(
        save_folder + save_filename,
        monitor='val_loss',
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
    use_multiprocessing=True
)

# %%
plt.rcParams.update({"savefig.facecolor": (1.0, 1.0, 1.0, 1)})  # disable transparent background
plt.tight_layout()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

if save_model and model_checkpoint:  # load best weights for test accuracy
    model_best = keras.models.load_model(save_folder + save_filename)
    print("Best:")
else:
    model_best = model
    print("Last:")
    if save_model:
        model.save(save_folder + save_filename)

# get model predictions
yhat_test = np.squeeze(model_best.predict(X_test))
yhat_train = np.squeeze(model_best.predict(X_train))

# print model predictions
print("Predictions (Test):")
for p, prediction in enumerate(yhat_test):
    print('{:.1f}\t'.format(prediction), end='')
    if p % N_examples_test == N_examples_test - 1:
        print('\t\t')
print("Predictions (Train):")
for p, prediction in enumerate(yhat_train):
    print('{:.1f}\t'.format(prediction), end='')
    if p % N_examples_train == N_examples_train - 1:
        print('\t\t')

# calculate result metrics
mu_test = np.zeros_like(d_all_labels, dtype=float)
std_test = np.zeros_like(d_all_labels, dtype=float)
mu_train = np.zeros_like(d_all_labels, dtype=float)
std_train = np.zeros_like(d_all_labels, dtype=float)

for d_index, d in enumerate(d_all_labels):
    yhat_test_d = yhat_test[y_test == d]
    mu_test[d_index] = np.mean(yhat_test_d)
    std_test[d_index] = np.std(yhat_test_d)

    yhat_train_d = yhat_train[y_train == d]
    mu_train[d_index] = np.mean(yhat_train_d)
    std_train[d_index] = np.std(yhat_train_d)

# for printing
df = DataFrame({"d": d_all_labels,
                "mu_test": mu_test,
                "std_test": std_test,
                # "ci_down_test": mu_test - 2 * std_test,
                # "ci_up_test": mu_test + 2 * std_test,
                "mu_train": mu_train,
                "std_train": std_train,
                # "ci_down_train": mu_train - 2 * std_train,
                # "ci_up_train": mu_train + 2 * std_train
                })
print(df.round(1).to_string(index=False))

if save_results:
    np.savetxt(save_folder + save_filename + '/y_test.txt', y_test)
    np.savetxt(save_folder + save_filename + '/yhat_test.txt', yhat_test)
    np.savetxt(save_folder + save_filename + '/y_train.txt', y_train)
    np.savetxt(save_folder + save_filename + '/yhat_train.txt', yhat_train)
    df.round(1).to_csv(save_folder + save_filename + '.csv', index=False)
    plt.savefig(save_folder + save_filename + '/plot_training.png')

plt.show()

# %%
