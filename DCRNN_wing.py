# %%
"""
Created on Wed Jun 13 17:52:13 2018

@author: yxf118
"""

from pathlib import Path
import os
from c_k_rnn import c_k_RNNCell
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# Parameters
learning_rate = 0.01
# batch_size = 40  # mini-batch size
epochs = 500
n_hidden = 256   # hidden unit numbers
n_layers = 1  # number of hidden layers used
keep_prob = 1.0  # Dropout keep probability
c_n = 1  # c_k_rnn, k parameter
records = np.zeros([epochs, 4])  # records keeping, nn_loss, reg_loss, train_acc, test_acc

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.05.25/filtered_a5_s10_o60/'  # include trailing slash
file_names = ['0-1', '6-1', '0-3', '6-3', '0-4', '6-4', '0-5', '6-5', '0-6', '6-6', '0-7', '6-7', '0-8', '6-8', '0-10', '6-10', '0-11', '6-11']
file_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
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
    file_names_test = ['0-9', '6-9']
    file_labels_test = [0, 1]
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 0.8
    shuffle_examples = True

# conv_filters = len(inputs_ft) + len(inputs_ang)
# conv_kernel_size = 1
# lstm_units = 128  # number of lstm cells of each lstm layer
# lr = 0.0001  # learning rate
# epochs_number = 1000  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped

save_plot = True
save_cm = True  # save confusion matrix
save_model = True  # save model file
save_folder = 'plots/2021.06.07_dcrnn/'  # include trailing slash
save_filename = root_folder + save_folder + ','.join(file_names) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_' + str(n_layers) + 'd' + str(n_hidden) + 'k' + str(c_n) + '_' + str(learning_rate)  # + '_f5,10,60'

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
        data[((k*N_examples + i) * N_per_example):((k*N_examples + i + 1) * N_per_example), :N_inputs_ft] = \
            ft_meas[inputs_ft, (i*N_per_step):(i*N_per_step + N_per_example)].T  # measured FT
        if N_inputs_ang > 0:
            data[((k*N_examples + i) * N_per_example):((k*N_examples + i + 1) * N_per_example), N_inputs_ft:] = \
                ang_meas[inputs_ang, (i*N_per_step):(i*N_per_step + N_per_example)].T  # stroke angle
        labels[k*N_examples + i] = file_labels[k]
        # sanity checks for data: looked at 1st row of 1st file, last row of 1st file, first row of 2nd file,
        # last row of last file, to make sure all the data I needed was at the right place

# %%
# save the min and max values used for normalization of the data
Path(save_filename).mkdir(parents=True, exist_ok=True)  # make folder
np.savetxt(save_filename + '/data_min.txt', np.min(data, axis=0))
np.savetxt(save_filename + '/data_max.txt', np.max(data, axis=0))

data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # normalize
data = data.reshape(N_files_total * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
# data = data.transpose(0, 2, 1)  # feature major

if shuffle_examples:  # randomize order of data to be split into train and test sets
    permutation = list(np.random.permutation(N_files_total * N_examples))
    data = data[permutation]
    labels = labels[permutation]

labels = np.eye(N_classes)[labels]  # one-hot labels

# split data into training and testing sets
X_train = data[:N_files_train*N_examples_train]
y_train = labels[:N_files_train*N_examples_train]
X_test = data[N_files_train*N_examples_train:]
y_test = labels[N_files_train*N_examples_train:]

batch_size = N_files_test*N_examples_test  # mini-batch size
assert batch_size <= N_files_test*N_examples_test  # need this to evaluate test accuracy
print('Mini-batch size:', batch_size)

# %%


def get_batches(X, Y, N_per_example, batch_size):
    '''
    slice the mini-batches

    X: X_input, to be sliced
    N_per_example: num of steps (in time)
    n_inputs: input features
    n_classes: output classes
    '''
    n_batches = int(len(X) / batch_size)
    # keep only integer batches
    X = X[:int(batch_size * n_batches)]
    Y = Y[:int(batch_size * n_batches)]
    # reshape
    for n in range(0, X.shape[0], batch_size):
        # inputs
        x = X[n:n+batch_size, :, :]
        # targets
        y = Y[n:n+batch_size, :]
        yield x, y

# batches = get_batches(X_train, y_train, N_per_example, N_inputs)
# x, y = next(batches)


def build_inputs(num_steps, N_classes):
    '''
    building the input layer

    num_steps: number of time steps in each sequence (2nd dimension)
    '''
    inputs = tf.placeholder(tf.float32, shape=(None, num_steps, N_inputs), name='inputs')
    targets = tf.placeholder(tf.float32, shape=(None, N_classes), name='targets')

    # add the keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob


def build_rnn(rnn_size, num_layers, batch_size, num_steps, keep_prob, c_k):
    '''
    building the rnn layer

    keep_prob: dropout keep probability
    rnn_size: number of hidden units in rnn layer
    num_layers: number of rnn layers
    batch_size: batch_size

    '''
    # build an rnn unit
    cell = c_k_RNNCell(rnn_size, c_k)
    # cell = forget_cell(rnn_size, c_k)

    # adding dropout
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    # rnn2 = c_k_RNNCell(rnn_size)
    # drop2 = tf.contrib.rnn.DropoutWrapper(rnn2, output_keep_prob=keep_prob)
    # stack_rnn = [drop]
    # for _ in range(num_layers-1):
    #     stack_rnn.append(drop2)
    # # stack (changed in TF 1.2)
    # cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)

    # stack_rnn = []
    # for _ in range(num_layers):
    #     stack_rnn.append(cell)
    # # stack (changed in TF 1.2)
    # cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)
    # initial_state = cell.zero_state(int(batch_size/num_steps), tf.float32)
    # only used the fist layer of RNN

    return cell, initial_state


def build_output(rnn_output, in_size, out_size, c_k):
    '''
    building the output layer

    rnn_output: the output of the rnn layer
    in_size: rnn layer reshaped size
    out_size: softmax layer size

    '''
#    rnn_output = rnn_output[:,in_size*(c_k-1):]
    rnn_output = rnn_output[:, :in_size]

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    # compute logits
    logits = tf.matmul(rnn_output, softmax_w) + softmax_b

    # softmax return
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits


def build_loss(logits, targets, rnn_size, N_classes, coeff_a, c_k):
    '''
    compute loss using logits and targets

    logits: fully connected layer output（before softmax）
    targets: targets
    rnn_size: rnn_size
    N_classes: class size

    '''
#    # One-hot coding
#    y_one_hot = tf.one_hot(targets, N_classes)
#    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

#    regularizer = tf.constant(0, dtype=tf.float32, name="beta_reg")
    # Regularizer for coeffs to give desired eigenvalues
    eig_desired = tf.constant(.01, shape=[c_k*rnn_size], dtype=tf.float32)
    eig = tf.linalg.eigvalsh(coeff_a)
    beta_reg = tf.constant(1, name="beta_reg", dtype=tf.float32)
#    beta_reg = tf.get_variable("beta_reg", shape=[], initializer=tf.constant_initializer(value=1), trainable=True, dtype=tf.float32)
    regularizer = beta_reg * tf.norm(eig-eig_desired, ord='euclidean')

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets)
    loss = tf.reduce_mean(loss)

#    targets = tf.reshape(targets, [-1, N_classes])
#    loss = tf.nn.l2_loss(logits-targets)
    return loss, regularizer


def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
#    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.01)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


class ckRNN:

    def __init__(self, N_classes, batch_size=1000, num_steps=100,
                 rnn_size=100, num_layers=n_layers, learning_rate=0.001,
                 c_k=1, grad_clip=5, sampling=False):

        # if sampling is True，use SGD, only 1 sample
        if sampling:
            batch_size = num_steps * 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # input layer
        self.inputs, self.targets, self.keep_prob = build_inputs(num_steps, N_classes)

        # rnn layerfrom f_gate_cell import forget_cell

        cell, self.initial_state = build_rnn(rnn_size, num_layers, batch_size, N_per_example, self.keep_prob, c_k)

#        # one-hot coding for inputs
#        x_one_hot = tf.one_hot(self.inputs, N_classes)

        # running the RNN
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state

        # predicting the results
        self.prediction, self.logits = build_output(state, rnn_size, N_classes, c_k)

#        self.coeff_a = tf.constant([0])
        coeff_a_kernel = tf.reshape(outputs[rnn_size:, -1, :], [c_k, rnn_size])
        coeff_a_eye = tf.eye(rnn_size, batch_shape=[c_k])
        self.coeff_a = tf.reshape(tf.transpose(tf.einsum('ij,ijk->ijk', coeff_a_kernel, coeff_a_eye), perm=[1, 0, 2]), [rnn_size, rnn_size*c_k])
        self.coeff_a = tf.concat([self.coeff_a[:, :rnn_size]+outputs[:rnn_size, -1, :], self.coeff_a[:, rnn_size:]], 1)
        if (c_k > 1):
            self.coeff_a = tf.concat([self.coeff_a,
                                      tf.convert_to_tensor(np.kron(np.eye(c_k-1, M=c_k), np.eye(rnn_size)), dtype=tf.float32)], 0)

        # Loss and optimizer (with gradient clipping)
        self.loss_nn, self.regularizer = build_loss(self.logits, self.targets, rnn_size, N_classes, self.coeff_a, c_k)
        self.loss = self.loss_nn + self.regularizer

        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

        self.accuracy, self.accuracy_op = tf.metrics.accuracy(labels=tf.argmax(self.targets, 1),
                                                              predictions=tf.argmax(self.logits, 1))


model = ckRNN(N_classes, batch_size=batch_size, num_steps=N_per_example,
              rnn_size=n_hidden, num_layers=n_layers,
              learning_rate=learning_rate, c_k=c_n)
graph = tf.get_default_graph()

# %%
saver = tf.train.Saver(max_to_keep=1)
# for i in tqdm(range(8, 10), desc="\nTraining progress"):
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    counter = 0

    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        sess.run(tf.local_variables_initializer())
        for x, y in get_batches(X_train, y_train, N_per_example, batch_size):
            counter += 1
            # start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss_nn, batch_loss_reg, train_acc, new_state, _ = sess.run([model.loss_nn, model.regularizer,
                                                                               model.accuracy_op,
                                                                               model.final_state,
                                                                               model.optimizer], feed_dict=feed)
            # end = time.time()
            records[e, 0] = batch_loss_nn
            records[e, 1] = batch_loss_reg
            records[e, 2] = train_acc

            # control the print lines
        if (e+1) % 1 == 0:
            sess.run(tf.local_variables_initializer())
#            stream_vars = [i for i in tf.local_variables()]
#            print('[total, count]:',sess.run(stream_vars))  #accuracy metrics
#                print('\n',
#                      'Epoches: {}/{}... '.format(e+1, epochs),
#                      'Training Steps: {}... '.format(counter),
#                      'Training Loss: {:.4f}... '.format(batch_loss_nn),
#                      'Regularizer Loss: {:.4f}... '.format(batch_loss_reg),
#                      'Training Accuracy: {:.4f}... '.format(train_acc),
#                      '{:.4f} sec/batch'.format((end-start)))
#                print("\n beta_reg=",graph.get_tensor_by_name("beta_reg:0").eval())

            # get average test accuracy across batches
            test_counter = 0
            test_loss_avg = 0
            test_acc_avg = 0

            for X_test_rnn, y_test_rnn in get_batches(X_test, y_test, N_per_example, batch_size):
                feed = {model.inputs: X_test_rnn,
                        model.targets: y_test_rnn,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                test_loss, test_acc = sess.run([model.loss,
                                                model.accuracy_op],
                                               feed_dict=feed)

                test_loss_avg += (test_loss - test_loss_avg) / (test_counter + 1)
                test_acc_avg += (test_acc - test_acc_avg) / (test_counter + 1)
                test_counter += 1

            test_loss = test_loss_avg
            test_acc = test_acc_avg

            if (e+1) % 10 == 0:
                print('\n',
                      'Epochs: {}... '.format(e+1),
                      'Train Accuracy: {:4f}... '.format(train_acc),
                      #   'Test loss: {:.4f}... '.format(test_loss),
                      'Test Accuracy: {:4f}... '.format(test_acc))
            records[e, 3] = test_acc

            if test_acc >= np.max(records[:, 3]) and save_model:  # save latest highest test accuracy model
                saver.save(sess, save_filename + "/checkpoints/e{}_train_acc={:.1f}_test_acc={:.1f}".format(e, train_acc*100, test_acc*100))
            # saver.save(sess, "./rnn_model/checkpoints/i{}_l{}.ckpt".format(counter, n_hidden))

# np.save('results_200/records_timesteps_{}_k_{}_trial_{}'.format(N_per_example,c_n,i),records)

# %%
plt.plot(records[:, 2])  # train accuracy
plt.plot(records[:, 3])  # test accuracy
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

if save_plot:
    np.save(save_filename + '/records', records)
    plt.savefig(save_filename + '.png')

plt.show()

# %%test


# checkpoint = tf.train.latest_checkpoint('./rnn_model/checkpoints/')
# model = ckRNN(y_train.shape[1], batch_size=batch_size, num_steps=N_per_example,
#                rnn_size=n_hidden, num_layers=1,
#                learning_rate=learning_rate, c_k=c_n)
# saver = tf.train.Saver()
# with tf.Session() as sess:
#    # load the model and restoring
#    saver.restore(sess, checkpoint)
#    new_state = sess.run(model.initial_state)
#    sess.run(tf.local_variables_initializer())
#    for X_test_rnn, y_test_rnn in get_batches(X_test, y_test, N_per_example, batch_size):
#        feed = {model.inputs: X_test_rnn,
#                model.targets: y_test_rnn,
#                model.keep_prob: 1.,
#                model.initial_state: new_state}
#        test_loss, test_acc = sess.run([model.loss,
#                                        model.accuracy_op],
#                                        feed_dict=feed)

#    print('\n',
#          'Test loss: {:.4f}... '.format(test_loss),
#          'Test Accuracy: {:4f}... '.format(test_acc))
