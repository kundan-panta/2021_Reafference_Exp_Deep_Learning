# %%
# python == 3.8.7
# torch == 1.9.0+cu111
# numpy == 1.19.3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
assert torch.cuda.is_available()
device = torch.device("cuda")

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.05.25/filtered_a5_s10_o60/'  # include trailing slash
file_names = ['0-1', '6-1', '12-1', '18-1', '24-1', '0-3', '6-3', '12-3', '18-3', '24-3', '0-4', '6-4', '12-4', '18-4', '24-4', '0-5', '6-5', '12-5', '18-5', '24-5', '0-6', '6-6', '12-6', '18-6', '24-6', '0-7', '6-7', '12-7', '18-7', '24-7', '0-8', '6-8', '12-8', '18-8', '24-8', '0-10', '6-10', '12-10', '18-10', '24-10', '0-11', '6-11', '12-11', '18-11', '24-11']
file_labels = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
trajectory_name = '30deg'  # choose trajectory name for which to process data

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 20

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]

# empirical_prediction = True  # whether to use collected data as the "perfect prediction"
# empirical_prediction_name = '22'
# subract_prediction = False  # meas - pred?

separate_test_files = True  # if using a separate set of files for testing
if separate_test_files:
    file_names_test = ['0-9', '6-9', '12-9', '18-9', '24-9']
    file_labels_test = [0, 0, 0, 1, 1]
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 0.8
    shuffle_examples = True

# conv_filters = len(inputs_ft) + len(inputs_ang)
# conv_kernel_size = 1
lstm_units = 256  # number of lstm cells of each lstm layer
lr = 0.001  # learning rate
epochs_number = 1000  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped
mini_batch_size = 50
num_layers = 4
dropout = 0

save_plot = True
save_cm = True  # save confusion matrix
save_model = True  # save model file
save_folder = 'plots/2021.06.29_pytorch/'  # include trailing slash
# save_filename = root_folder + save_folder + ','.join(file_names) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_' + str(num_layers) + 'r' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
save_filename = root_folder + save_folder + 'all_' + ','.join(str(temp) for temp in file_labels_test) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_' + str(num_layers) + 'r' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'

# %%
# all files to extract the data from
N_files_train = len(file_names)
if separate_test_files:  # add test files to the list
    N_files_test = len(file_names_test)
    file_names.extend(file_names_test)
    file_labels.extend(file_labels_test)
else:
    N_files_test = N_files_train
N_files_total = len(file_names)

assert len(file_labels) == N_files_total  # makes sure labels are there for all files

# get stroke cycle period information from one of the files
t = np.around(np.loadtxt(root_folder + data_folder + file_names[0] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
cpg_param = np.loadtxt(root_folder + data_folder + file_names[0] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

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
data = np.zeros((N_files_total * N_examples * N_per_example, N_inputs))  # all input data
labels = np.zeros((N_files_total * N_examples), dtype=int)  # all labels

# if empirical_prediction:  # furthest distance from wall as forward model
#     ft_pred = np.loadtxt(root_folder + data_folder + empirical_prediction_name + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)

for k in range(N_files_total):
    # get data
    t = np.around(np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    # if not(empirical_prediction):  # use QS model if empirical prediction is not used
    #     ft_pred = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_pred.csv', delimiter=',', unpack=True)
    ft_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(root_folder + data_folder + file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)

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
data = data.reshape(N_files_total * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
# data = data.transpose(1, 0, 2)  # sequence -> batches -> features

if shuffle_examples:  # randomize order of data to be split into train and test sets
    permutation = list(np.random.permutation(N_files_total * N_examples))
    data = data[permutation]
    labels = labels[permutation]

# labels = np.eye(N_classes)[labels]  # one-hot labels

# split data into training and testing sets
x = data[:N_files_train * N_examples_train]
y = labels[:N_files_train * N_examples_train]
x_val = data[N_files_train * N_examples_train:]
y_val = labels[N_files_train * N_examples_train:]

# %%
x = torch.from_numpy(x).float().to(device=device)
y = torch.from_numpy(y).long().to(device=device)
x_val = torch.from_numpy(x_val).float().to(device=device)
y_val = torch.from_numpy(y_val).long().to(device=device)

# # %%
# x_mini = torch.utils.data.DataLoader(x, batch_size=mini_batch_size, shuffle=False)
# y_mini = torch.utils.data.DataLoader(y, batch_size=mini_batch_size, shuffle=False)
# x_val_mini = torch.utils.data.DataLoader(x_val, batch_size=mini_batch_size, shuffle=False)
# y_val_mini = torch.utils.data.DataLoader(y_val, batch_size=mini_batch_size, shuffle=False)

# %%
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/


class RNNModelCustom(torch.nn.Module):
    def __init__(self):
        super(RNNModelCustom, self).__init__()

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = torch.nn.RNN(input_size=N_inputs,
                                hidden_size=lstm_units,
                                num_layers=num_layers,
                                nonlinearity='relu',
                                batch_first=True,
                                dropout=dropout)

        # Readout layer
        self.fc = torch.nn.Linear(lstm_units, N_classes)

    def forward(self, x):
        batch_size = x.shape[0]  # variable batch size
        # Initialize hidden state with zeros
        hn = torch.zeros((num_layers, batch_size, lstm_units), device=device).float().requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, hn.detach())

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


model = RNNModelCustom()
model.to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(model)
for param in list(model.parameters()):
    print(param.size())

# %%
for epoch in range(epochs_number):
    model.train()  # train mode

    # shuffle the training data each epoch
    permutation = list(np.random.permutation(N_files_train * N_examples_train))
    x = x[permutation]
    y = y[permutation]

    for mini_batch in range(0, N_files_train * N_examples_train, mini_batch_size):
        # do I need to make x_mini a copy of the slice???
        x_mini = x[mini_batch:mini_batch + mini_batch_size].requires_grad_()
        y_mini = y[mini_batch:mini_batch + mini_batch_size]

        model.zero_grad()

        outputs = model(x_mini)

        loss = criterion(outputs, y_mini)
        loss.backward()

        optimizer.step()

    # print accuracy info
    if (epoch + 1) % 10 == 0:
        model.eval()  # test mode

        # train
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        accuracy_train = float((predicted == y).sum()) / x.shape[0] * 100

        # test
        outputs = model(x_val)
        _, predicted = torch.max(outputs.data, 1)
        accuracy_test = float((predicted == y_val).sum()) / x_val.shape[0] * 100

        print('Epoch: {}. Train: {:.2f}%. Test: {:.2f}%'.format(epoch + 1, accuracy_train, accuracy_test))

# %%
# plt.rcParams.update({"savefig.facecolor": (1.0, 1.0, 1.0, 1)})  # disable transparent background
# plt.tight_layout()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')

# # make and save the confusion matrix twice
# cm_train = confusion_matrix(y, np.argmax(model.predict(x), axis=-1))
# cm_test = confusion_matrix(y_val, np.argmax(model.predict(x_val), axis=-1))
# if save_cm:
#     np.savetxt(save_filename + '/cm_train_last.txt', cm_train, fmt='%d')
#     np.savetxt(save_filename + '/cm_test_last.txt', cm_test, fmt='%d')
# print('Last:')
# print('Train accuracy: {:.1f}%\tTest accuracy: {:.1f}%'.format(np.trace(cm_train) / np.sum(cm_train) * 100, np.trace(cm_test) / np.sum(cm_test) * 100))
# print(cm_train)
# print(cm_test)
# # print(model.predict(x_val))

# if save_model:  # load best weights for test accuracy
#     model = keras.models.load_model(save_filename)
#     # confusion matrix again for best test weights
#     cm_train = confusion_matrix(y, np.argmax(model.predict(x), axis=-1))
#     cm_test = confusion_matrix(y_val, np.argmax(model.predict(x_val), axis=-1))
#     if save_cm:
#         np.savetxt(save_filename + '/cm_train_best.txt', cm_train, fmt='%d')
#         np.savetxt(save_filename + '/cm_test_best.txt', cm_test, fmt='%d')
#     print('Best:')
#     print('Train accuracy: {:.1f}%\tTest accuracy: {:.1f}%'.format(np.trace(cm_train) / np.sum(cm_train) * 100, np.trace(cm_test) / np.sum(cm_test) * 100))
#     print(cm_train)
#     print(cm_test)
#     # print(model.predict(x_val))

# if save_plot:
#     plt.savefig(save_filename + '.png')
# plt.show()

# %%
