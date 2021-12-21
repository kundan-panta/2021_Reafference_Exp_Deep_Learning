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
    for s_index, s in enumerate(sets_train):
        for d_index, d in enumerate(d_train[s_index]):
            file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
            file_labels_train.append(d_train_labels[s_index][d_index])

    file_names_val = []
    file_labels_val = []
    for s_index, s in enumerate(sets_val):
        for d_index, d in enumerate(d_val[s_index]):
            file_names_val.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, d))
            file_labels_val.append(d_val_labels[s_index][d_index])

    file_names = file_names_train + file_names_val
    file_labels = file_labels_train + file_labels_val

    # baseline file names for each set
    if baseline_d is not None:
        baseline_file_names_train = []
        for s_index, s in enumerate(sets_train):
            for d_index, d in enumerate(d_train[s_index]):
                baseline_file_names_train.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

        baseline_file_names_val = []
        for s_index, s in enumerate(sets_val):
            for d_index, d in enumerate(d_val[s_index]):
                baseline_file_names_val.append('Ro={}/A={}/Set={}/d={}'.format(str(Ro), str(A_star), s, baseline_d))

        baseline_file_names = baseline_file_names_train + baseline_file_names_val
        assert len(baseline_file_names) == len(file_names)

    return file_names, file_labels,\
        file_names_train, file_labels_train,\
        file_names_val, file_labels_val,\
        baseline_file_names_train, baseline_file_names_val, baseline_file_names


def data_get_info(data_folder,
                  file_names, file_labels,
                  file_names_train, file_names_val,
                  train_val_split, separate_val_files,
                  N_cycles_example, N_cycles_step, N_cycles_to_use,
                  inputs_ft, inputs_ang):
    import numpy as np

    # %%
    N_files_train = len(file_names_train)
    N_files_val = len(file_names_val)
    if not(separate_val_files):  # if separate test files are not provided, then we use all the files for both training and testing
        N_files_val = N_files_train
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
    print('Unused data points:', N_total - ((N_examples - 1) * N_per_step + N_per_example))  # print number of unused data points
    print('Total examples per file:', N_examples)
    print('Training examples per file:', N_examples_train)
    print('Validation examples per file:', N_examples_val)
    print('Inputs:', N_inputs)
    # print('Clases:', N_classes)

    return N_files_all, N_files_train, N_files_val,\
        N_examples, N_examples_train, N_examples_val,\
        N_per_example, N_per_step, N_total,\
        N_inputs, N_inputs_ft, N_inputs_ang


def data_load(data_folder,
              file_names, file_labels,
              baseline_d, baseline_file_names,
              inputs_ft, inputs_ang,
              N_files_all, N_examples, N_per_example, N_per_step, N_inputs, N_inputs_ft, N_inputs_ang):
    import numpy as np

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

    return data, labels


def data_process(data, labels,
                 separate_val_files, shuffle_examples, shuffle_seed,
                 save_model, save_results, save_folder, save_filename,
                 N_files_all, N_files_train, N_examples, N_examples_train, N_examples_val, N_per_example, N_inputs):
    import numpy as np
    from pathlib import Path

    # %%
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # save the min and max values used for normalization of the data
    if save_model or save_results:
        Path(save_folder + save_filename).mkdir(parents=True, exist_ok=True)  # make folder
    if save_model:
        np.savetxt(save_folder + save_filename + '/data_min.txt', data_min)
        np.savetxt(save_folder + save_filename + '/data_max.txt', data_max)

    data = (data - data_min) / (data_max - data_min)  # normalize
    data = data.reshape(N_files_all * N_examples, N_per_example, N_inputs)  # example -> all data points of that example -> FT components
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
        data = data[permutation]
        labels = labels[permutation]

    # labels = np.eye(N_classes)[labels]  # one-hot labels

    # split data into training and testing sets
    X_train = data[:N_files_train * N_examples_train]
    y_train = labels[:N_files_train * N_examples_train]
    X_val = data[N_files_train * N_examples_train:]
    y_val = labels[N_files_train * N_examples_train:]

    return X_train, y_train, X_val, y_val


def model_build_tf(lstm_units, epochs_patience, lr,
                   save_model, model_checkpoint, save_folder, save_filename,
                   N_per_example, N_inputs):
    from tensorflow import keras

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
            keras.layers.Dense(lstm_units, activation='relu'),  # , activation='elu'),
            keras.layers.Dense(1, activation='relu')  # , activation='exponential')
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


def model_fit_tf(model, callbacks_list, epochs_number,
                 X_train, y_train, X_val, y_val):
    # %%
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_number,
        verbose=0,
        callbacks=callbacks_list,
        shuffle=True,
        workers=1,
        use_multiprocessing=False
    )

    return history


def model_predict_tf(model, save_model, model_checkpoint, save_folder, save_filename,
                     X_train, X_val):
    import numpy as np
    from tensorflow import keras

    # %% predict distance to wall
    if save_model and model_checkpoint:  # load best weights for test accuracy
        model_best = keras.models.load_model(save_folder + save_filename)
        print("Best:")
    else:
        model_best = model
        print("Last:")
        if save_model:
            model.save(save_folder + save_filename)

    # get model predictions
    yhat_train = np.squeeze(model_best.predict(X_train))
    yhat_val = np.squeeze(model_best.predict(X_val))

    return yhat_train, yhat_val


def model_evaluate_regression_tf(history,
                                 y_train, y_val, yhat_train, yhat_val,
                                 save_results, save_folder, save_filename,
                                 file_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import DataFrame

    # %% evaluate performance
    # calculate result metrics
    d_all_labels = np.unique(file_labels)
    mu_train = np.zeros_like(d_all_labels, dtype=float)
    std_train = np.zeros_like(d_all_labels, dtype=float)
    mu_val = np.zeros_like(d_all_labels, dtype=float)
    std_val = np.zeros_like(d_all_labels, dtype=float)

    for d_index, d in enumerate(d_all_labels):
        yhat_train_d = yhat_train[y_train == d]
        mu_train[d_index] = np.mean(yhat_train_d)
        std_train[d_index] = np.std(yhat_train_d)

        yhat_val_d = yhat_val[y_val == d]
        mu_val[d_index] = np.mean(yhat_val_d)
        std_val[d_index] = np.std(yhat_val_d)

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
                    "mu_val": mu_val,
                    "std_val": std_val,
                    # "ci_down_val": mu_val - 2 * std_val,
                    # "ci_up_val": mu_val + 2 * std_val,
                    "mu_train": mu_train,
                    "std_train": std_train,
                    # "ci_down_train": mu_train - 2 * std_train,
                    # "ci_up_train": mu_train + 2 * std_train
                    })
    print(df.round(1).to_string(index=False))

    # %% plot performance plot as well
    plt.rcParams.update({"savefig.facecolor": (1, 1, 1, 1)})  # disable transparent background
    plt.rc('font', family='serif', size=12)
    plt.tight_layout()

    # for testing data
    fig_yhat_val = plt.figure(figsize=(4, 4))

    plt.plot(d_all_labels, mu_val - 2 * std_val, 'r--')
    plt.plot(d_all_labels, mu_val + 2 * std_val, 'r--')
    plt.fill_between(d_all_labels, mu_val - 2 * std_val, mu_val + 2 * std_val, color='r', alpha=.2)
    plt.plot(d_all_labels, mu_val, 'bo--', label='Predicted')
    plt.plot([np.min(d_all_labels), np.max(d_all_labels)], [np.min(d_all_labels), np.max(d_all_labels)], 'k-', label='Actual')

    plt.xlabel('True Distance (cm)')
    plt.ylabel('Distance to Wall (cm)')
    plt.title('Test')
    plt.legend()

    plt.axhline(0, color='silver')  # x = 0
    plt.axvline(0, color='silver')  # y = 0
    plt.axis('square')
    plt.xlim(0, 50)
    plt.ylim(0, 50)

    # same for training data
    fig_yhat_train = plt.figure(figsize=(4, 4))

    plt.plot(d_all_labels, mu_train - 2 * std_train, 'r--')
    plt.plot(d_all_labels, mu_train + 2 * std_train, 'r--')
    plt.fill_between(d_all_labels, mu_train - 2 * std_train, mu_train + 2 * std_train, color='r', alpha=.2)
    plt.plot(d_all_labels, mu_train, 'bo--', label='Predicted')
    plt.plot([np.min(d_all_labels), np.max(d_all_labels)], [np.min(d_all_labels), np.max(d_all_labels)], 'k-', label='Actual')

    plt.xlabel('True Distance (cm)')
    plt.ylabel('Distance to Wall (cm)')
    plt.title('Train')
    plt.legend()

    plt.axhline(0, color='silver')  # x = 0
    plt.axvline(0, color='silver')  # y = 0
    plt.axis('square')
    plt.xlim(0, 50)
    plt.ylim(0, 50)

    # %%
    fig_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')

    # %%
    if save_results:
        np.savetxt(save_folder + save_filename + '/y_val.txt', y_val)
        np.savetxt(save_folder + save_filename + '/yhat_val.txt', yhat_val)
        np.savetxt(save_folder + save_filename + '/y_train.txt', y_train)
        np.savetxt(save_folder + save_filename + '/yhat_train.txt', yhat_train)
        df.round(1).to_csv(save_folder + save_filename + '/yhat_stats.csv', index=False)
        fig_yhat_train.savefig(save_folder + save_filename + '/plot_yhat_train.svg')
        fig_yhat_val.savefig(save_folder + save_filename + '.svg')
        fig_loss.savefig(save_folder + save_filename + '/plot_training.svg')

    plt.show()

    return df
