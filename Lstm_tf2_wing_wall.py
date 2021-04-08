# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from correct_biases import correct_biases
from tensorflow import keras
# %%
# all files to extract the data from (collected at multiple locations)
file_names = ['12', '30']
N_files = len(file_names)

# also convert the list into an array of floats
file_names_float = np.zeros(N_files)
for i in range(N_files):
    file_names_float[i] = float(file_names[i])

# choose trajectory name for which to process data
trajectory_name = '30deg'

# parameter to choose if biases should be corrected
biases = True

# %%
# for each file, I need 6 components of rms ft
data = np.zeros((N_files*4940, 6))
x = np.zeros((15*N_files, 6, 260))
y = np.zeros((15*N_files))
x_val = np.zeros((4*N_files, 6, 260))
y_val = np.zeros((4*N_files))
for k in range(N_files):
    # get data
    t = np.around(np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 't.csv', delimiter=',', unpack=True), decimals=3)  # round to ms
    ft_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_meas.csv', delimiter=',', unpack=True)
    ang_meas = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_meas.csv', delimiter=',', unpack=True)
    cpg_param = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'cpg_param.csv', delimiter=',', unpack=True)

    if biases:
        ft_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ft_bias.csv', delimiter=',', unpack=True)
        ang_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'ang_bias.csv', delimiter=',', unpack=True)
        gravity_bias = np.loadtxt(file_names[k] + '/' + trajectory_name + '/' + 'gravity_bias.csv', delimiter=',', unpack=True)

        # remove the three biases and rotate the frame to align with normally used frame
        ft_meas = correct_biases(ft_meas, ft_bias[:, 0], ang_bias[0], gravity_bias[:, 0])
    #ft_meas_norm = ((ft_meas.T-np.min(ft_meas,axis=1))/(np.max(ft_meas,axis=1)-np.min(ft_meas,axis=1))).T
    data[4940*k:4940*(k+1),:] = ft_meas[:,0:260*19].T
    
data = (data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))
ft_meas_norm = data.reshape(19*N_files,260,6)
ft_meas_norm = ft_meas_norm.transpose(0,2,1)    
for k in range(N_files):
    x[15*k:15*(k+1),:,:] = ft_meas_norm[19*k:19*(k+1)-4,:,:]
    y[15*k:15*(k+1)] = k;
    x_val[4*k:4*(k+1),:,:] = ft_meas_norm[19*k+15:19*(k+1),:,:]
    y_val[4*k:4*(k+1)] = k;
    
model = keras.models.Sequential(
    [
     keras.layers.RNN(keras.layers.LSTMCell(128), return_sequences=True, input_shape=(6, 260)),
     keras.layers.RNN(keras.layers.LSTMCell(128)),
     keras.layers.Dense(2),
      ]
    )


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)


history = model.fit(
    x, y, validation_data=(x_val, y_val), epochs=300
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()