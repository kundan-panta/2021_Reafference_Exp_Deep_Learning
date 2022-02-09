# %%
# python == 3.8.7
# tensorflow == 2.4.0
# numpy == 1.19.3

# %% turn into function
def experiment(parameters):
    root_folder, Ro, A_star, sets_val, sets_test, average_window, lstm_layers, dense_hidden_layers, N_units, lr, dropout = parameters

    # %%
    # from helper_functions import divide_file_names, data_get_info, data_load, data_shorten_sequence
    from helper_functions import data_full_process
    # from helper_functions import model_k_fold_tf
    from helper_functions import model_build_tf, model_fit_tf
    from helper_functions import model_predict_tf, model_evaluate_regression_tf
    # import matplotlib.pyplot as plt
    # import numpy as np
    # %load_ext autoreload
    # %autoreload 2

    # %% design parameters
    # root_folder = ''  # include trailing slash
    data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
    # Ro = 3.5
    # A_star = 2
    Ro_d_last = {2: 46, 3.5: 43, 5: 40}  # furthest distance from wall for each wing shape

    # all sets except the ones given in sets_val
    sets_train = [1, 2, 3, 4, 5]
    [sets_train.remove(set_val) for set_val in sets_val]
    [sets_train.remove(set_test) for set_test in sets_test]

    # sets_train = [1, 2, 3, 4]
    d_train = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_train)  # list of all distances from wall for each set
    d_train_labels = d_train

    # sets_val = [3]
    d_val = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_val)  # list of all distances from wall
    d_val_labels = d_val

    # sets_test = [5]
    d_test = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_test)  # list of all distances from wall
    d_test_labels = d_test

    separate_val_files = len(sets_val) > 0
    if separate_val_files:
        train_val_split = 1
        shuffle_examples = False
        shuffle_seed = None
    else:
        train_val_split = 0.8
        shuffle_examples = True
        shuffle_seed = 5  # seed to split data in reproducible way

    N_cycles_example = 1  # use this number of stroke cycles as 1 example
    N_cycles_step = 1  # number of cycles to step between consecutive examples
    # total number of cycles to use per file
    # set 0 to automatically calculate number of examples from the first file
    N_cycles_to_use = 14

    inputs_ft = [0, 1, 2, 3, 4, 5]
    inputs_ang = [0]
    # average_window = 10

    baseline_d = None  # set to None for no baseline

    # lstm_layers = 2
    # dense_hidden_layers = 1
    # N_units = 16  # number of lstm cells of each lstm layer
    # lr = 0.0002  # learning rate
    # dropout = 0.2
    recurrent_dropout = 0.0
    epochs_number = 10000  # number of epochs
    epochs_patience = 10000  # for early stopping, set <0 to disable
    k_fold_splits = len(sets_train)

    save_model = True  # save model file, save last model if model_checkpoint == False
    model_checkpoint = False  # doesn't do anything if save_model == False
    save_results = True
    save_folder = root_folder + 'plots/2022.02.09_experiment/'  # include trailing slash
    save_filename = 'Ro={}_A={}_Tr={}_Val={}_Te={}_in={}_bl={}_Ne={}_Ns={}_win={}_{}L{}D{}_lr={}_dr={}_recdr={}'.format(
        Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
        ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft),
        baseline_d, N_cycles_example, N_cycles_step, average_window,
        lstm_layers, dense_hidden_layers, N_units, lr, dropout, recurrent_dropout)

    # %% load the data
    X_train, y_train, X_val, y_val, N_per_example, N_inputs, file_labels, data_min, data_max = \
        data_full_process(sets_train, d_train, d_train_labels,
                          sets_val, d_val, d_val_labels,
                          baseline_d,
                          Ro, A_star,
                          inputs_ft, inputs_ang,
                          None, None,  # no info on min and max data for normalizaton
                          N_cycles_example, N_cycles_step, N_cycles_to_use, average_window,
                          train_val_split, separate_val_files, shuffle_examples, shuffle_seed,
                          data_folder, save_model, save_results, save_folder, save_filename)

    # %% initialize the model
    model, callbacks_list = \
        model_build_tf(lstm_layers, dense_hidden_layers, N_units,
                       epochs_patience, lr, dropout, recurrent_dropout,
                       save_model, model_checkpoint,
                       save_folder, save_filename,
                       N_per_example, N_inputs)

    # %% train the model
    model, history = \
        model_fit_tf(model, callbacks_list, epochs_number,
                     X_train, y_train, X_val, y_val)

    # %% train the model using k-fold CV
    # model, history = \
    #     model_k_fold_tf(X_train, y_train,
    #                     lstm_layers, dense_hidden_layers, N_units,
    #                     lr, epochs_number, epochs_patience, dropout, recurrent_dropout,
    #                     k_fold_splits, shuffle_seed,
    #                     save_model, model_checkpoint, save_results,
    #                     save_folder, save_filename,
    #                     N_per_example, N_inputs, file_labels)

    # %% load actual test data
    X_test, y_test, _, _, _, _, _, _, _ = \
        data_full_process(sets_test, d_test, d_test_labels,
                          [], [], [],
                          baseline_d,
                          Ro, A_star,
                          inputs_ft, inputs_ang,
                          data_min, data_max,
                          N_cycles_example, N_cycles_step, N_cycles_to_use, average_window,
                          1, True, False, None,
                          data_folder, False, False, '', '')

    # %% predict on training and testing data using trained model
    yhat_train, yhat_val = \
        model_predict_tf(model,
                         save_model, model_checkpoint,
                         save_folder, save_filename,
                         X_train, X_val)

    # %% evaluate performance
    df_val, loss_val_all = \
        model_evaluate_regression_tf(history,
                                     y_train, y_val, yhat_train, yhat_val,
                                     save_results, save_folder, save_filename, 'val',
                                     file_labels)

    # %% predict on training and testing data using trained model
    yhat_train, yhat_test = \
        model_predict_tf(model,
                         save_model, model_checkpoint,
                         save_folder, save_filename,
                         X_train, X_test)

    # %% evaluate performance
    df_test, loss_test_all = \
        model_evaluate_regression_tf(history,
                                     y_train, y_test, yhat_train, yhat_test,
                                     save_results, save_folder, save_filename, 'test',
                                     file_labels)

# %%
