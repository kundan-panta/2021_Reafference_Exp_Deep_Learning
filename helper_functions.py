# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from pandas import DataFrame


def divide_file_names(sets_train, d_train, d_train_labels,
                      sets_val, d_val, d_val_labels,
                      baseline_d,
                      Ro, A_star):

    # test that the sets and distances are assigned correctly
    assert len(sets_train) == len(d_train)
    for i in range(len(sets_train)):
        assert len(d_train[i]) == len(d_train_labels[i])

    assert len(sets_val) == len(d_val)
    for i in range(len(sets_val)):
        assert len(d_val[i]) == len(d_val_labels[i])

    # get the file names and labels
    file_names_train = []
    file_labels_train = []
    file_sets_train = []
    for s_index, s in enumerate(sets_train):
        for d_index, d in enumerate(d_train[s_index]):
            file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
            file_labels_train.append(d_train_labels[s_index][d_index])
            file_sets_train.append(s)

    file_names_val = []
    file_labels_val = []
    file_sets_val = []
    for s_index, s in enumerate(sets_val):
        for d_index, d in enumerate(d_val[s_index]):
            file_names_val.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
            file_labels_val.append(d_val_labels[s_index][d_index])
            file_sets_val.append(s)

    file_names = file_names_train + file_names_val
    file_labels = file_labels_train + file_labels_val
    file_sets = file_sets_train + file_sets_val

    # baseline file names for each set
    baseline_file_names_train = []
    baseline_file_names_val = []
    baseline_file_names = []

    if baseline_d is not None:
        for s_index, s in enumerate(sets_train):
            for d_index, d in enumerate(d_train[s_index]):
                baseline_file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

        for s_index, s in enumerate(sets_val):
            for d_index, d in enumerate(d_val[s_index]):
                baseline_file_names_val.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

        baseline_file_names = baseline_file_names_train + baseline_file_names_val
        assert len(baseline_file_names) == len(file_names)

    return file_names, file_labels, file_sets,\
        file_names_train, file_labels_train, file_sets_train,\
        file_names_val, file_labels_val, file_sets_val,\
        baseline_file_names_train, baseline_file_names_val, baseline_file_names


def data_get_info(data_folder,
                  file_names, file_labels,
                  file_names_train, file_names_val,
                  train_val_split, separate_val_files,
                  N_cycles_example, N_cycles_step, N_cycles_to_use,
                  inputs_ft, inputs_ang):

    # %%
    N_files_train = len(file_names_train)
    N_files_val = len(file_names_val)
    if not(separate_val_files):  # if separate test files are not provided, then we use all the files for both training and testing
        N_files_val = N_files_train
    N_files_all = len(file_names)

    assert len(file_labels) == N_files_all  # makes sure labels are there for all files

    # get stroke cycle period information from one of the files
    t = np.around(np.loadtxt(data_folder + file_names[0] + '/' + 't.csv', delimiter=','), decimals=3)  # round to ms
    cpg_param = np.loadtxt(data_folder + file_names[0] + '/' + 'cpg_param.csv', delimiter=',')
    ang_meas = np.loadtxt(data_folder + file_names[0] + '/' + 'ang_meas.csv', delimiter=',')

    t_s = round(t[1] - t[0], 3)  # sample time
    freq = cpg_param[0, -1]  # store frequency of param set
    t_cycle = 1 / freq  # stroke cycle time

    # if N_cycles_to_use == 0:  # if number of cycles per file is not explicitly specified
    #     N_total = len(t)  # number of data points
    # else:
    #     N_per_cycle = round(t_cycle / t_s)  # number of data points per cycle, round instead of floor
    #     N_total = N_cycles_to_use * N_per_cycle + 100  # limit amount of data to use

    N_total = len(t)  # number of data points
    N_per_cycle = round(t_cycle / t_s)  # number of data points per cycle, round instead of floor

    N_per_example = round(N_cycles_example * t_cycle / t_s)  # number of data points per example, round instead of floor
    N_per_step = round(N_cycles_step * t_cycle / t_s)

    # set the phase of each example
    # find indices where stroke angle is 0 exactly or it changes signs
    # https://stackoverflow.com/questions/61233411/find-indices-where-a-python-array-becomes-positive-but-not-negative
    zero_ind = np.where(ang_meas[:-1, 0] * ang_meas[1:, 0] <= 0)[0]
    zero_ind = zero_ind[ang_meas[zero_ind, 0] <= 0]
    zero_ind = zero_ind[ang_meas[zero_ind + 20, 0] > 0]  # make sure no accidental + to - changes
    zero_ind = zero_ind[np.append(np.diff(zero_ind), -1) > N_per_cycle - 20]  # remove indices that are less than a N_per_step apart

    # start at a different phase
    # zero_ind += N_per_cycle // 4

    if zero_ind[-1] + N_per_example > N_total:
        zero_ind = zero_ind[:-1]  # remove last index if not enough data points
    print("Available cycles:", len(zero_ind))

    # update number of examples
    if N_cycles_to_use == 0:
        N_cycles_to_use = len(zero_ind)
    else:
        zero_ind = zero_ind[len(zero_ind) - N_cycles_to_use:]  # skip the 1st cycles if too many cycles available

    N_examples = (N_total - zero_ind[0] - N_per_example) // N_per_step + 1  # floor division
    assert N_total - zero_ind[0] >= (N_examples - 1) * N_per_step + N_per_example  # last data point used must not exceed total number of data points

    # number of training and testing stroke cycles
    N_examples_train = round(train_val_split * N_examples)
    if separate_val_files:
        N_examples_val = N_examples
    else:
        N_examples_val = N_examples - N_examples_train

    N_inputs_ft = len(inputs_ft)
    N_inputs_ang = len(inputs_ang)
    N_inputs = N_inputs_ft + N_inputs_ang  # ft_meas + other inputs

    # N_classes = len(np.unique(file_labels))
    # assert np.max(file_labels) == N_classes - 1  # check for missing labels in between

    print('Frequency:', freq)
    print('Data points in an example:', N_per_example)
    # print('Unused data points:', N_total - ((N_examples - 1) * N_per_step + N_per_example))  # print number of unused data points
    print('Total examples per file:', N_examples)
    print('Training examples per file:', N_examples_train)
    print('Validation examples per file:', N_examples_val)
    print('Inputs:', N_inputs)
    # print('Clases:', N_classes)

    return N_files_all, N_files_train, N_files_val,\
        N_examples, N_examples_train, N_examples_val,\
        N_per_example, N_per_step, N_total, zero_ind,\
        N_inputs, N_inputs_ft, N_inputs_ang


def data_load(data_folder,
              file_names, file_labels,
              baseline_d, baseline_file_names,
              inputs_ft, inputs_ang,
              separate_val_files, shuffle_examples, shuffle_seed,
              save_model, save_results, save_folder, save_filename,
              N_files_all, N_files_train,
              N_examples, N_examples_train, N_examples_val,
              N_cycles_step, N_per_example, N_total, zero_ind,
              N_inputs, N_inputs_ft, N_inputs_ang,
              data_min, data_max):

    # %%
    data = np.zeros((N_files_all, N_total, N_inputs))  # all input data
    # labels = np.zeros((N_files_all * N_examples))  # , dtype=int)  # all labels

    for k in range(N_files_all):
        # get data
        t = np.around(np.loadtxt(data_folder + file_names[k] + '/' + 't.csv', delimiter=','), decimals=3)  # round to ms
        ft_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ft_meas.csv', delimiter=',')
        if N_inputs_ang > 0:
            ang_meas = np.loadtxt(data_folder + file_names[k] + '/' + 'ang_meas.csv', delimiter=',')

        if baseline_d is not None:  # subtract pred from meas?
            baseline_ft_meas = np.loadtxt(data_folder + baseline_file_names[k] + '/' + 'ft_meas.csv', delimiter=',')
            ft_meas -= baseline_ft_meas

        # for i in range(N_examples):
        #     data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), :N_inputs_ft] = \
        #         ft_meas[inputs_ft, (i * N_per_step):(i * N_per_step + N_per_example)].T  # measured FT
        #     if N_inputs_ang > 0:
        #         data[((k * N_examples + i) * N_per_example):((k * N_examples + i + 1) * N_per_example), N_inputs_ft:] = \
        #             ang_meas[inputs_ang, (i * N_per_step):(i * N_per_step + N_per_example)].T  # stroke angle
        #     labels[k * N_examples + i] = file_labels[k]
        #     # sanity checks for data: looked at 1st row of 1st file, last row of 1st file, first row of 2nd file,
        #     # last row of last file, to make sure all the data I needed was at the right place

        data[k, :, :N_inputs_ft] = ft_meas[:, inputs_ft]
        if N_inputs_ang > 0:
            data[k, :, N_inputs_ft:] = ang_meas[:, inputs_ang]

    # %% convert the data into examples
    # make a new array with examples as the 1st dim
    X_all = np.zeros((N_files_all * N_examples, N_per_example, N_inputs))
    y_all = np.zeros((N_files_all * N_examples))  # , dtype=int)  # all labels

    for k in range(N_files_all):
        for i in range(N_examples):
            X_all[k * N_examples + i] = data[k, zero_ind[i * N_cycles_step]: zero_ind[i * N_cycles_step] + N_per_example, :]
        y_all[k * N_examples: (k + 1) * N_examples] = file_labels[k]

    # %%
    if (data_min is None) or (data_max is None):
        data_min = np.min(data, axis=(0, 1))
        data_max = np.max(data, axis=(0, 1))

    # save the min and max values used for normalization of the data
    if save_model or save_results:
        Path(save_folder + save_filename).mkdir(parents=True, exist_ok=True)  # make folder
    if save_model:
        np.savetxt(save_folder + save_filename + '/data_min.txt', data_min)
        np.savetxt(save_folder + save_filename + '/data_max.txt', data_max)

    # normalize
    data_min_reshaped = np.reshape(data_min, (1, 1, -1))
    data_max_reshaped = np.reshape(data_max, (1, 1, -1))
    X_all = (X_all - data_min_reshaped) / (data_max_reshaped - data_min_reshaped)

    # data = (data - data_min) / (data_max - data_min)  # normalize
    # data = data.reshape(N_files_all * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
    # data = data.transpose(0, 2, 1)  # feature major

    if shuffle_examples:  # randomize order of data to be split into train and test sets
        if not(separate_val_files):
            # shuffle every N_examples examples
            # then pick the first N_examples_train examples and put it to training set
            # and the remaining (N_examples_val) examples into the testing set
            N_examples_train_all = N_files_train * N_examples_train
            permutation = np.zeros(N_files_all * N_examples, dtype=int)
            for k in range(N_files_all):  # each file has N_example examples, and everything is in order
                shuffled = np.array(np.random.RandomState(seed=shuffle_seed + k).permutation(N_examples), dtype=int)
                permutation[k * N_examples_train:(k + 1) * N_examples_train] = k * N_examples + shuffled[:N_examples_train]
                permutation[N_examples_train_all + k * N_examples_val:N_examples_train_all + (k + 1) * N_examples_val] = k * N_examples + shuffled[N_examples_train:]
        else:
            permutation = list(np.random.RandomState(seed=shuffle_seed).permutation(N_files_all * N_examples))
        X_all = X_all[permutation]
        y_all = y_all[permutation]

    # labels = np.eye(N_classes)[labels]  # one-hot labels

    # split data into training and testing sets
    X_train = X_all[:N_files_train * N_examples_train]
    y_train = y_all[:N_files_train * N_examples_train]
    X_val = X_all[N_files_train * N_examples_train:]
    y_val = y_all[N_files_train * N_examples_train:]

    return X_train, y_train, X_val, y_val, data_min, data_max


def data_shorten_sequence(X_train, X_val, N_per_example, average_window):
    # %% reduce sequence length
    N_per_example = N_per_example // average_window  # update sequence length

    # cut out last data points so the number of data points is divisible by average_window
    X_train = X_train[:, 0:N_per_example * average_window, :]
    X_val = X_val[:, 0:N_per_example * average_window, :]

    # reshape the time series so mean can be taken for consecutive average_window time steps
    X_train = X_train.reshape(X_train.shape[0], -1, average_window, X_train.shape[2]).mean(axis=2)
    if X_val.size > 0:
        X_val = X_val.reshape(X_val.shape[0], -1, average_window, X_val.shape[2]).mean(axis=2)

    print('Data points in an example after averaging:', N_per_example)

    return X_train, X_val, N_per_example


def model_lstm_tf(lstm_layers, dense_hidden_layers, N_units,
                  dropout, recurrent_dropout, N_per_example, N_inputs):

    model = keras.models.Sequential()  # initialize

    # LSTM layers
    if lstm_layers == 1:
        model.add(keras.layers.LSTM(N_units, input_shape=(N_per_example, N_inputs), recurrent_dropout=recurrent_dropout))
    else:
        # first LSTM layer
        model.add(keras.layers.LSTM(N_units, input_shape=(N_per_example, N_inputs), recurrent_dropout=recurrent_dropout, return_sequences=True))
        # middle LSTM layers
        for _ in range(lstm_layers - 2):
            model.add(keras.layers.LSTM(N_units, recurrent_dropout=recurrent_dropout, dropout=dropout, return_sequences=True))
        # final LSTM layer
        model.add(keras.layers.LSTM(N_units, recurrent_dropout=recurrent_dropout, dropout=dropout))

    # Dense hidden layers
    for _ in range(dense_hidden_layers):
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(N_units, activation='elu'))

    # Output layer
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))

    # model = keras.models.Sequential(
    #     [
    #         # keras.layers.Conv1D(conv_filters, conv_kernel_size, activation='relu', input_shape=(N_per_example, N_inputs)),
    #         # keras.layers.Conv1D(N_inputs, 3, activation='relu'),
    #         keras.layers.LSTM(N_units, return_sequences=True, input_shape=(N_per_example, N_inputs), recurrent_dropout=recurrent_dropout),
    #         keras.layers.LSTM(N_units, recurrent_dropout=recurrent_dropout, dropout=dropout),
    #         # keras.layers.GRU(N_units, return_sequences=True, input_shape=(N_per_example, N_inputs)),
    #         # keras.layers.GRU(N_units),
    #         # keras.layers.RNN(keras.layers.LSTMCell(N_units), return_sequences=True, input_shape=(N_per_example, N_inputs)),
    #         # keras.layers.RNN(keras.layers.LSTMCell(N_units)),
    #         # keras.layers.SimpleRNN(N_units, return_sequences=True, input_shape=(N_per_example, N_inputs), unroll=True),
    #         # keras.layers.SimpleRNN(N_units),
    #         keras.layers.Dropout(dropout),
    #         keras.layers.Dense(N_units, activation='elu'),
    #         keras.layers.Dropout(dropout),
    #         keras.layers.Dense(1)  # , activation='relu')  # , activation='exponential')
    #     ]
    # )

    return model


def model_build_tf(lstm_layers, dense_hidden_layers, N_units,
                   epochs_patience, lr, dropout, recurrent_dropout,
                   save_model, model_checkpoint, save_results,
                   save_folder, save_filename,
                   N_per_example, N_inputs):
    # %%
    keras.backend.clear_session()

    model = model_lstm_tf(lstm_layers, dense_hidden_layers, N_units,
                          dropout, recurrent_dropout, N_per_example, N_inputs)

    model.compile(
        loss=keras.losses.LogCosh(),
        optimizer="adam",
        # metrics=["accuracy"],
        # steps_per_execution=100
    )
    keras.backend.set_value(model.optimizer.learning_rate, lr)
    print("Learning rate:", model.optimizer.learning_rate.numpy())

    def print_summary(info):
        # function to print and save model info
        if save_model:
            Path(save_folder + save_filename).mkdir(parents=True, exist_ok=True)  # make folder
            f = open(save_folder + save_filename + '/model_info.txt', 'a')
            f.write(info)
            f.write('\n')
            f.close()
        print(info)
    model.summary(print_fn=print_summary)

    # callbacks
    callbacks_list = []

    if epochs_patience > -1:
        early_stopping_monitor = early_stopping_custom_tf(
            monitor='val_loss',
            mode='auto',
            min_delta=0,
            patience=epochs_patience,
            baseline=None,
            restore_best_weights=True,
            verbose=0,
            save_info=save_results,
            save_filename=save_folder + save_filename
        )
        callbacks_list.append(early_stopping_monitor)

    if save_model and model_checkpoint:
        model_checkpoint_monitor = keras.callbacks.ModelCheckpoint(
            save_folder + save_filename,
            monitor='val_loss',
            mode='auto',
            save_best_only=True,
            verbose=0,
            save_freq="epoch"
        )
        callbacks_list.append(model_checkpoint_monitor)

    return model, callbacks_list


def model_fit_tf(model, callbacks_list, epochs_number,
                 X_train, y_train, X_val, y_val):

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_number,
        verbose=0,
        callbacks=callbacks_list,
        shuffle=True,
        workers=4,
        use_multiprocessing=True
    )

    return model, history


def model_predict_tf(model, save_model, model_checkpoint, save_folder, save_filename,
                     X_train, X_val):

    # %% predict distance to wall
    if save_model and model_checkpoint:  # load best weights for test accuracy
        model = keras.models.load_model(save_folder + save_filename)
        # print("Best:")
    else:
        # model = model
        # print("Last:")
        if save_model:
            model.save(save_folder + save_filename)

    # get model predictions
    yhat_train = np.squeeze(model.predict(X_train))
    yhat_val = np.squeeze(model.predict(X_val))

    return yhat_train, yhat_val


def model_evaluate_regression_tf(history,
                                 y_train, y_val, yhat_train, yhat_val,
                                 save_results, save_folder, save_filename, test_or_val,
                                 file_labels):

    # %% evaluate performance
    # calculate result metrics
    d_all_labels = np.unique(file_labels)
    mu_train = np.zeros_like(d_all_labels, dtype=float)
    std_train = np.zeros_like(d_all_labels, dtype=float)
    loss_mu_train = np.zeros_like(d_all_labels, dtype=float)
    loss_std_train = np.zeros_like(d_all_labels, dtype=float)
    mu_val = np.zeros_like(d_all_labels, dtype=float)
    std_val = np.zeros_like(d_all_labels, dtype=float)
    loss_mu_val = np.zeros_like(d_all_labels, dtype=float)
    loss_std_val = np.zeros_like(d_all_labels, dtype=float)

    def loss_fn(y, yhat):
        # return loss per sample
        # NOTE: input arrays need to be reshaped into 2D
        loss_fn_all = keras.losses.LogCosh(reduction='none')
        return loss_fn_all(np.reshape(y, (-1, 1)), np.reshape(yhat, (-1, 1))).numpy()

    for d_index, d in enumerate(d_all_labels):
        yhat_train_d = yhat_train[y_train == d]
        mu_train[d_index] = np.mean(yhat_train_d)
        std_train[d_index] = np.std(yhat_train_d)
        loss_train_d = loss_fn(y_train[y_train == d], yhat_train_d)
        loss_mu_train[d_index] = np.mean(loss_train_d)
        loss_std_train[d_index] = np.std(loss_train_d)

        yhat_val_d = yhat_val[y_val == d]
        mu_val[d_index] = np.mean(yhat_val_d)
        std_val[d_index] = np.std(yhat_val_d)
        loss_val_d = loss_fn(y_val[y_val == d], yhat_val_d)
        loss_mu_val[d_index] = np.mean(loss_val_d)
        loss_std_val[d_index] = np.std(loss_val_d)

    loss_train_all = loss_fn(y_train, yhat_train)
    loss_val_all = loss_fn(y_val, yhat_val)
    # print("Average", test_or_val, "loss:", loss_val_all)

    # %% print model predictions
    # print("Predictions (Test):")
    # for p, prediction in enumerate(yhat_val):
    #     print('{:.1f}\t'.format(prediction), end='')
    #     if p % N_examples_val == N_examples_val - 1:
    #         print('\t\t')
    # print("Predictions (Train):")
    # for p, prediction in enumerate(yhat_train):
    #     print('{:.1f}\t'.format(prediction), end='')
    #     if p % N_examples_train == N_examples_train - 1:
    #         print('\t\t')

    # for printing
    df = DataFrame({"d": d_all_labels,
                    "mu_{}".format(test_or_val): mu_val,
                    "std_{}".format(test_or_val): std_val,
                    "loss_mu_{}".format(test_or_val): loss_mu_val,
                    "loss_std_{}".format(test_or_val): loss_std_val,
                    # "ci_down_val": mu_val - 2 * std_val,
                    # "ci_up_val": mu_val + 2 * std_val,
                    "mu_train": mu_train,
                    "std_train": std_train,
                    "loss_mu_train": loss_mu_train,
                    "loss_std_train": loss_std_train,
                    # "ci_down_train": mu_train - 2 * std_train,
                    # "ci_up_train": mu_train + 2 * std_train
                    })
    # print(df.round(1).to_string(index=False))

    # %% plot performance plot as well
    plt.rcParams.update({"savefig.facecolor": (1, 1, 1, 1)})  # disable transparent background
    plt.rc('font', family='serif', size=12)
    plt.tight_layout()

    # for training data
    fig_yhat_train = plt.figure(figsize=(4, 4))

    plt.plot(d_all_labels, mu_train - 2 * std_train, 'r--')
    plt.plot(d_all_labels, mu_train + 2 * std_train, 'r--')
    plt.fill_between(d_all_labels, mu_train - 2 * std_train, mu_train + 2 * std_train, color='r', alpha=.2)
    plt.plot(d_all_labels, mu_train, 'bo--', label='Predicted')
    plt.plot([np.min(d_all_labels), np.max(d_all_labels)], [np.min(d_all_labels), np.max(d_all_labels)], 'k-', label='Actual')

    plt.xlabel('True Distance (cm)')
    plt.ylabel('Predicted Distance (cm)')
    plt.title('Train')
    plt.legend()

    plt.axhline(0, color='silver')  # x = 0
    plt.axvline(0, color='silver')  # y = 0
    plt.axis('square')
    plt.xlim(0, 50)
    plt.ylim(0, 50)

    # for val data
    fig_yhat_val = plt.figure(figsize=(4, 4))

    plt.plot(d_all_labels, mu_val - 2 * std_val, 'r--')
    plt.plot(d_all_labels, mu_val + 2 * std_val, 'r--')
    plt.fill_between(d_all_labels, mu_val - 2 * std_val, mu_val + 2 * std_val, color='r', alpha=.2)
    plt.plot(d_all_labels, mu_val, 'bo--', label='Predicted')
    plt.plot([np.min(d_all_labels), np.max(d_all_labels)], [np.min(d_all_labels), np.max(d_all_labels)], 'k-', label='Actual')

    plt.xlabel('True Distance (cm)')
    plt.ylabel('Predicted Distance (cm)')
    plt.title(test_or_val.title())
    plt.legend()

    plt.axhline(0, color='silver')  # x = 0
    plt.axvline(0, color='silver')  # y = 0
    plt.axis('square')
    plt.xlim(0, 50)
    plt.ylim(0, 50)

    # loss over distance
    fig_loss_dist = plt.figure(figsize=(4, 4))

    plt.errorbar(d_all_labels, loss_mu_train, fmt='bo--', label='Train', yerr=2*loss_std_train, capsize=4)
    plt.errorbar(d_all_labels, loss_mu_val, fmt='rx:', label=test_or_val.title(), yerr=2*loss_std_val, capsize=4)

    plt.xlabel('True Distance (cm)')
    plt.ylabel('Loss')
    plt.legend()

    plt.xlim(0, 50)

    # loss over time
    fig_loss_training = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', test_or_val.title()], loc='best')

    # %%
    if save_results:
        np.savetxt(save_folder + save_filename + '/y_{}.txt'.format(test_or_val), y_val)
        np.savetxt(save_folder + save_filename + '/yhat_{}.txt'.format(test_or_val), yhat_val)
        np.savetxt(save_folder + save_filename + '/y_train.txt', y_train)
        np.savetxt(save_folder + save_filename + '/yhat_train.txt', yhat_train)
        np.savetxt(save_folder + save_filename + '/loss_{}_all.txt'.format(test_or_val), loss_val_all)
        np.savetxt(save_folder + save_filename + '/loss_train_all.txt', loss_train_all)
        df.to_csv(save_folder + save_filename + '/yhat_stats_{}.csv'.format(test_or_val), index=False)

        fig_yhat_train.savefig(save_folder + save_filename + '/plot_yhat_train.svg')
        fig_loss_dist.savefig(save_folder + save_filename + '/plot_loss_{}_dist.svg'.format(test_or_val))

        if test_or_val == 'test':
            fig_yhat_val.savefig(save_folder + save_filename + '.svg')
        else:
            fig_yhat_val.savefig(save_folder + save_filename + '/plot_yhat_val.svg')
            fig_loss_training.savefig(save_folder + save_filename + '/plot_training.svg')
            np.save(save_folder + save_filename + '/history.npy', history.history)
            # load the history dictionary with:
            # history = np.load(save_folder + save_filename + '/history.npy', allow_pickle='TRUE').item()

    # plt.show()
    plt.close(fig_yhat_train)
    plt.close(fig_yhat_val)
    plt.close(fig_loss_dist)
    plt.close(fig_loss_training)

    return df, np.mean(loss_val_all)


def model_k_fold_tf(X_train, y_train,
                    lstm_layers, dense_hidden_layers, N_units,
                    lr, epochs_number, epochs_patience, dropout, recurrent_dropout,
                    k_fold_splits, shuffle_seed,
                    save_model, model_checkpoint, save_results,
                    save_folder, save_filename,
                    N_per_example, N_inputs, file_labels
                    ):
    # https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k_fold_splits, shuffle=False, random_state=shuffle_seed)

    VALIDATION_LOSS = [np.inf] * k_fold_splits
    models = [0] * k_fold_splits
    histories = [0] * k_fold_splits
    fold_var = 0

    for train_index, val_index in kf.split(X_train, y_train):
        # get training and validation data
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]

        # build the model
        save_filename_fold = save_filename + '/fold={}'.format(fold_var)
        models[fold_var], callbacks_list = \
            model_build_tf(lstm_layers, dense_hidden_layers, N_units,
                           epochs_patience, lr, dropout, recurrent_dropout,
                           save_model, model_checkpoint, save_results,
                           save_folder, save_filename_fold,
                           N_per_example, N_inputs)

        # FIT THE MODEL
        models[fold_var], histories[fold_var] = \
            model_fit_tf(models[fold_var], callbacks_list, epochs_number,
                         X_train, y_train, X_val_fold, y_val_fold)

        # get validation performance for model selection
        yhat_train_fold, yhat_val_fold = \
            model_predict_tf(models[fold_var], save_model, model_checkpoint, save_folder, save_filename_fold,
                             X_train_fold, X_val_fold)

        df_fold, loss_val_all_fold = \
            model_evaluate_regression_tf(histories[fold_var],
                                         y_train_fold, y_val_fold, yhat_train_fold, yhat_val_fold,
                                         save_results, save_folder, save_filename_fold,
                                         file_labels)

        VALIDATION_LOSS[fold_var] = loss_val_all_fold

        # keras.backend.clear_session()

        fold_var += 1

    # best model
    fold_best = np.argmin(VALIDATION_LOSS)
    # print(VALIDATION_LOSS)
    # print(models)

    if save_model or save_results:
        np.savetxt(save_folder + save_filename + '/fold_best.txt', fold_best, fmt='%i')
        # f = open(save_folder + save_filename + '/fold_best.txt', 'w')
        # f.write(str(fold_best))
        # f.close()
        # save_filename_fold = save_filename + '/fold={}'.format(fold_best)
        # model = keras.models.load_model(save_folder + save_filename_fold)

    return models[fold_best], histories[fold_best]


def early_stopping_custom_tf(monitor='val_loss',
                             mode='auto',
                             min_delta=0,
                             patience=0,
                             baseline=None,
                             restore_best_weights=True,
                             verbose=0,
                             save_info=False,
                             save_filename=None):
    # Modified to restore best weights at the end of training, even if early stopping was not triggered
    # Base code from https://github.com/keras-team/keras/blob/v2.7.0/keras/callbacks.py
    # Latest commit 9088756 on Sep 17, 2021

    # import numpy as np
    from tensorflow.keras.callbacks import Callback
    from tensorflow.python.platform import tf_logging as logging

    class EarlyStoppingCustom(Callback):
        """Stop training when a monitored metric has stopped improving.

        Assuming the goal of a training is to minimize the loss. With this, the
        metric to be monitored would be `'loss'`, and mode would be `'min'`. A
        `model.fit()` training loop will check at end of every epoch whether
        the loss is no longer decreasing, considering the `min_delta` and
        `patience` if applicable. Once it's found no longer decreasing,
        `model.stop_training` is marked True and the training terminates.

        The quantity to be monitored needs to be available in `logs` dict.
        To make it so, pass the loss or metrics at `model.compile()`.

        Args:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity
            monitored has stopped increasing; in `"auto"`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used. An epoch will be restored regardless
            of the performance relative to the `baseline`. If no epoch
            improves on `baseline`, training will run for `patience`
            epochs and restore weights from the best epoch in that set.

        Example:

        >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        >>> # This callback will stop the training when there is no improvement in
        >>> # the loss for three consecutive epochs.
        >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
        >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
        >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
        ...                     epochs=10, batch_size=1, callbacks=[callback],
        ...                     verbose=0)
        >>> len(history.history['loss'])  # Only 4 epochs are run.
        4
        """

        def __init__(self,
                     monitor='val_loss',
                     min_delta=0,
                     patience=0,
                     verbose=0,
                     mode='auto',
                     baseline=None,
                     restore_best_weights=False):
            super(EarlyStoppingCustom, self).__init__()

            self.monitor = monitor
            self.patience = patience
            self.verbose = verbose
            self.baseline = baseline
            self.min_delta = abs(min_delta)
            self.wait = 0
            self.stopped_epoch = 0
            self.restore_best_weights = restore_best_weights
            self.best_weights = None

            if mode not in ['auto', 'min', 'max']:
                logging.warning('EarlyStopping mode %s is unknown, '
                                'fallback to auto mode.', mode)
                mode = 'auto'

            if mode == 'min':
                self.monitor_op = np.less
            elif mode == 'max':
                self.monitor_op = np.greater
            else:
                if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
                        self.monitor.endswith('auc')):
                    self.monitor_op = np.greater
                else:
                    self.monitor_op = np.less

            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1

        def on_train_begin(self, logs=None):
            # Allow instances to be re-used
            self.wait = 0
            self.stopped_epoch = 0
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.best_weights = None
            self.best_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.restore_best_weights and self.best_weights is None:
                # Restore the weights after first epoch if no progress is ever made.
                self.best_weights = self.model.get_weights()

            self.wait += 1
            if self._is_improvement(current, self.best):
                self.best = current
                self.best_epoch = epoch
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                # Only restart wait if we beat both the baseline and our previous best.
                if self.baseline is None or self._is_improvement(current, self.baseline):
                    self.wait = 0

            # Only check after the first epoch.
            if self.wait >= self.patience and epoch > 0:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # if self.restore_best_weights and self.best_weights is not None:
                #     if self.verbose > 0:
                #         print('Restoring model weights from the end of the best epoch: '
                #               f'{self.best_epoch + 1}.')
                #     self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0 and self.verbose > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

            if self.restore_best_weights and self.best_weights is not None:
                # if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch: '
                      f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

            if save_info:
                np.savetxt(save_filename + '/epoch_best.txt', [self.best_epoch + 1], fmt='%i')

        def get_monitor_value(self, logs):
            logs = logs or {}
            monitor_value = logs.get(self.monitor)
            if monitor_value is None:
                logging.warning('Early stopping conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))
            return monitor_value

        def _is_improvement(self, monitor_value, reference_value):
            return self.monitor_op(monitor_value - self.min_delta, reference_value)

    return EarlyStoppingCustom(
        monitor=monitor,
        mode=mode,
        min_delta=min_delta,
        patience=patience,
        baseline=baseline,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )


def data_full_process(sets_train, d_train, d_train_labels,
                      sets_val, d_val, d_val_labels,
                      baseline_d,
                      Ro, A_star,
                      inputs_ft, inputs_ang,
                      data_min, data_max,
                      N_cycles_example, N_cycles_step, N_cycles_to_use, average_window,
                      train_val_split, separate_val_files, shuffle_examples, shuffle_seed,
                      data_folder, save_model, save_results, save_folder, save_filename):

    # %% get the file names to load data from
    file_names, file_labels, file_sets,\
        file_names_train, file_labels_train, file_sets_train,\
        file_names_val, file_labels_val, file_sets_val,\
        baseline_file_names_train, baseline_file_names_val, baseline_file_names = \
        divide_file_names(sets_train, d_train, d_train_labels,
                          sets_val, d_val, d_val_labels,
                          baseline_d,
                          Ro, A_star)

    # %% get info about the data
    N_files_all, N_files_train, N_files_val,\
        N_examples, N_examples_train, N_examples_val,\
        N_per_example, N_per_step, N_total, zero_ind,\
        N_inputs, N_inputs_ft, N_inputs_ang = \
        data_get_info(data_folder,
                      file_names, file_labels,
                      file_names_train, file_names_val,
                      train_val_split, separate_val_files,
                      N_cycles_example, N_cycles_step, N_cycles_to_use,
                      inputs_ft, inputs_ang)

    # %% get training and validation datasets
    X_train, y_train, X_val, y_val, data_min, data_max = \
        data_load(data_folder,
                  file_names, file_labels,
                  baseline_d, baseline_file_names,
                  inputs_ft, inputs_ang,
                  separate_val_files, shuffle_examples, shuffle_seed,
                  save_model, save_results, save_folder, save_filename,
                  N_files_all, N_files_train,
                  N_examples, N_examples_train, N_examples_val,
                  N_cycles_step, N_per_example, N_total, zero_ind,
                  N_inputs, N_inputs_ft, N_inputs_ang,
                  data_min, data_max)

    # %% reduce sequence length
    X_train, X_val, N_per_example = \
        data_shorten_sequence(X_train, X_val, N_per_example, average_window)

    return X_train, y_train, X_val, y_val, N_per_example, N_inputs, file_labels, data_min, data_max


# Transformer stuff
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    from tensorflow.keras import layers

    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_return_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def model_build_transformer_tf(lstm_layers, N_units, epochs_patience, lr,
                               save_model, model_checkpoint, save_folder, save_filename,
                               N_per_example, N_inputs):
    input_shape = (N_per_example, N_inputs)

    model = transformer_return_model(
        input_shape,
        head_size=4,
        num_heads=4,
        ff_dim=4,  # PROBABLY CHANGE THIS TO 1 OR GET RID OF THE CONV1D LAYER
        num_transformer_blocks=lstm_layers,
        mlp_units=[N_units],
        mlp_dropout=0,
        dropout=0,
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer="adam",
        # metrics=["accuracy"],
        # steps_per_execution=100
    )
    keras.backend.set_value(model.optimizer.learning_rate, lr)
    print("Learning rate:", model.optimizer.learning_rate.numpy())

    model.summary()

    # callbacks
    callbacks_list = []

    if epochs_patience > -1:
        early_stopping_monitor = keras.callbacks.EarlyStopping(
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
        model_checkpoint_monitor = keras.callbacks.ModelCheckpoint(
            save_folder + save_filename,
            monitor='val_loss',
            mode='auto',
            save_best_only=True,
            verbose=0
        )
        callbacks_list.append(model_checkpoint_monitor)

    return model, callbacks_list
