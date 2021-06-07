##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:52:13 2018

@author: yxf118
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import time
from c_k_rnn import c_k_RNNCell

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = 'data/2021.05.25/filtered_a5_s10_o60/'  # include trailing slash
file_names = ['18-1', '24-1', '18-3', '24-3', '18-4', '24-4', '18-5', '24-5', '18-6', '24-6', '18-7', '24-7', '18-8', '24-8', '18-10', '24-10', '18-11', '24-11']
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
    file_names_test = ['18-9', '24-9']
    file_labels_test = [0, 1]
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 0.8
    shuffle_examples = True

# conv_filters = len(inputs_ft) + len(inputs_ang)
# conv_kernel_size = 1
lstm_units = 128  # number of lstm cells of each lstm layer
lr = 0.0001  # learning rate
epochs_number = 1000  # number of epochs
# epochs_patience = 400  # number of epochs of no improvement after which training is stopped

save_plot = True
save_cm = True  # save confusion matrix
save_model = True  # save model file
save_folder = 'plots/2021.06.06_rnn/'  # include trailing slash
save_filename = root_folder + save_folder + ','.join(file_names) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2r' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'

# %%
# Parameters
learning_rate = 1e-3
batch_size = 1000
epochs = 1000
n_hidden = 128   # hidden unit numbers
n_layers = 1 #number of hidden layers used
keep_prob = 1.0 # Dropout keep probability
c_n = 4 #c_k_rnn, k parameter
records = np.zeros([epochs,4]) #records keeping, nn_loss, reg_loss, train_acc, test_acc

#Lorenz Attractor data
data = np.load('data_10.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
n_input = 3         #data feature diemnsion
n_steps = X_train.shape[1]       #timesteps used in RNN
num_classes = y_train.shape[1]


##mnist data
#mnist = tf.keras.datasets.mnist
#(X_train, y_train),(X_test, y_test) = mnist.load_data()
#X_train, X_test = X_train / 255.0, X_test / 255.0
#y_train = np.eye(10)[y_train]
#y_test = np.eye(10)[y_test]
#n_input = 28         #data feature diemnsion
#n_steps = 28       #timesteps used in RNN
#num_classes = y_train.shape[1]



def get_batches(X, Y, n_steps, batch_size):
    '''
    slice the mini-batches
    
    X: X_input, to be sliced
    n_steps: num of steps (in time)
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
        x = X[n:n+batch_size,:,:]
        # targets
        y = Y[n:n+batch_size,:]
        yield x,y
        
#batches = get_batches(X_train, y_train, n_steps, n_input)
#x, y = next(batches)

       
def build_inputs(num_steps, num_classes):
    '''
    building the input layer
    
    num_steps: number of time steps in each sequence (2nd dimension)
    '''
    inputs = tf.placeholder(tf.float32, shape=(None, num_steps, n_input), name='inputs')
    targets = tf.placeholder(tf.float32, shape=(None, num_classes), name='targets')

    # add the keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob


def build_rnn(rnn_size, num_layers, batch_size,num_steps, keep_prob, c_k):
    ''' 
    building the rnn layer
        
    keep_prob: dropout keep probability
    rnn_size: number of hidden units in rnn layer
    num_layers: number of rnn layers
    batch_size: batch_size

    '''
    # build an rnn unit
    cell = c_k_RNNCell(rnn_size, c_k)
#    cell = forget_cell(rnn_size, c_k)
    
    # adding dropout
#    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
#    rnn2 = c_k_RNNCell(rnn_size)
#    drop2 = tf.contrib.rnn.DropoutWrapper(rnn2, output_keep_prob=keep_prob)
#    stack_rnn = [drop]
#    for _ in range(num_layers-1):
#        stack_rnn.append(drop2)
#    # stack (changed in TF 1.2)   
#    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
     
    initial_state = cell.zero_state(batch_size, tf.float32)
#    initial_state = cell.zero_state(int(batch_size/num_steps), tf.float32)
    #only used the fist layer of RNN
    
    return cell, initial_state


def build_output(rnn_output , in_size, out_size, c_k):
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


def build_loss(logits, targets, rnn_size, num_classes, coeff_a, c_k):
    '''
    compute loss using logits and targets
    
    logits: fully connected layer output（before softmax）
    targets: targets
    rnn_size: rnn_size
    num_classes: class size
        
    '''
#    # One-hot coding
#    y_one_hot = tf.one_hot(targets, num_classes)
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
    
#    targets = tf.reshape(targets, [-1, num_classes])
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
    
    def __init__(self, num_classes, batch_size=1000, num_steps=100, 
                       rnn_size=100, num_layers=n_layers, learning_rate=0.001, 
                       c_k=1, grad_clip=5, sampling=False):
    
        # if sampling is True，use SGD, only 1 sample
        if sampling == True:
            batch_size = num_steps * 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # input layer
        self.inputs, self.targets, self.keep_prob = build_inputs(num_steps, num_classes)

        # rnn layerfrom f_gate_cell import forget_cell

        cell, self.initial_state = build_rnn(rnn_size, num_layers, batch_size, n_steps, self.keep_prob, c_k)

#        # one-hot coding for inputs
#        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # running the RNN
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state
        
        # predicting the results
        self.prediction, self.logits = build_output(state, rnn_size, num_classes, c_k)

#        self.coeff_a = tf.constant([0])
        coeff_a_kernel = tf.reshape(outputs[rnn_size: , -1, : ], [c_k, rnn_size])
        coeff_a_eye = tf.eye(rnn_size, batch_shape=[c_k])
        self.coeff_a= tf.reshape(tf.transpose(tf.einsum('ij,ijk->ijk',coeff_a_kernel,coeff_a_eye), perm=[1,0,2]), [rnn_size, rnn_size*c_k])
        self.coeff_a = tf.concat([self.coeff_a[:,:rnn_size]+outputs[:rnn_size, -1, :], self.coeff_a[:,rnn_size:]], 1)
        if (c_k > 1) :
            self.coeff_a = tf.concat([self.coeff_a, 
                                     tf.convert_to_tensor(np.kron(np.eye(c_k-1, M=c_k), np.eye(rnn_size)), dtype=tf.float32)], 0)
        
        # Loss and optimizer (with gradient clipping)
        self.loss_nn, self.regularizer = build_loss(self.logits, self.targets, rnn_size, num_classes, self.coeff_a, c_k)
        self.loss = self.loss_nn + self.regularizer
        
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(labels=tf.argmax(self.targets,1),
                                                              predictions=tf.argmax(self.logits,1))


model = ckRNN(num_classes, batch_size=batch_size, num_steps=n_steps,
                rnn_size=n_hidden, num_layers=1, 
                learning_rate=learning_rate, c_k=c_n)
graph = tf.get_default_graph()


saver = tf.train.Saver()
for i in tqdm(range(8,10), desc="\nTraining progress"):
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        counter = 0
            
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            sess.run(tf.local_variables_initializer())
            for x, y in get_batches(X_train, y_train, n_steps, batch_size):
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss_nn, batch_loss_reg, train_acc, new_state, _ = sess.run([model.loss_nn, model.regularizer,
                                                                                   model.accuracy_op,
                                                                                   model.final_state, 
                                                                                   model.optimizer], feed_dict=feed)
                end = time.time()
                records[e,0] = batch_loss_nn
                records[e,1] = batch_loss_reg
                records[e,2] = train_acc
                
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
    
                for X_test_rnn, y_test_rnn in get_batches(X_test, y_test, 
                                                          n_steps, batch_size):
                    feed = {model.inputs: X_test_rnn,
                            model.targets: y_test_rnn,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    test_loss, test_acc = sess.run([model.loss,
                                                    model.accuracy_op], 
                                                    feed_dict=feed)
                    
                if (e+1) % 10 == 0:
                    print('\n', 
                          'Test loss: {:.4f}... '.format(test_loss),
                          'Test Accuracy: {:4f}... '.format(test_acc),
                          'Epochs: {}'.format(e+1))
                records[e,3] = test_acc
    
    #        saver.save(sess, "./rnn_model/checkpoints/i{}_l{}.ckpt".format(counter, n_hidden))
    
#    np.save('results_200/records_timesteps_{}_k_{}_trial_{}'.format(n_steps,c_n,i),records)
    np.save('results/records_timesteps_{}_k_{}_trial_{}'.format(n_steps,c_n,i),records)

##%%test
#       
#
#checkpoint = tf.train.latest_checkpoint('./rnn_model/checkpoints/')
#model = ckRNN(y_train.shape[1], batch_size=batch_size, num_steps=n_steps,
#                rnn_size=n_hidden, num_layers=1, 
#                learning_rate=learning_rate, c_k=c_n)
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    # load the model and restoring
#    saver.restore(sess, checkpoint)
#    new_state = sess.run(model.initial_state)
#    sess.run(tf.local_variables_initializer())
#    for X_test_rnn, y_test_rnn in get_batches(X_test, y_test, n_steps, batch_size):
#        feed = {model.inputs: X_test_rnn,
#                model.targets: y_test_rnn,
#                model.keep_prob: 1.,
#                model.initial_state: new_state}
#        test_loss, test_acc = sess.run([model.loss,
#                                        model.accuracy_op], 
#                                        feed_dict=feed)
#
#    print('\n', 
#          'Test loss: {:.4f}... '.format(test_loss),
#          'Test Accuracy: {:4f}... '.format(test_acc))