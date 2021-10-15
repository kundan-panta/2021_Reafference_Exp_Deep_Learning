# %%
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/butterworth_h0.04_l5_o10/'  # include trailing slash
Ro = 3.5
A_star = 2

# sets_train = [1, 2, 3, 4, 5]
# d_train = [list(range(1, 43 + 1, 3))] * 5  # list of all distances from wall for each set
# d_train_labels = d_train

sets_test = []
d_test = []  # list of all distances from wall
d_test_labels = d_test

separate_test_files = len(sets_test) > 0
if separate_test_files:
    train_test_split = 1
    shuffle_examples = False
else:
    train_test_split = 1.0
    shuffle_examples = True
    shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 0

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = []

baseline_d = None  # set to None for no baseline

lstm_units = 64  # number of lstm cells of each lstm layer
lr = 0.0001  # learning rate
epochs_number = 1500  # number of epochs
epochs_patience = -1  # for early stopping, set <0 to disable

save_model = False  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2021.10.14_new plot code/'  # include trailing slash
# save_filename = ','.join(file_names_train) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2l' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
# save_filename = 'all_' + ','.join(str(temp) for temp in file_labels_test) + '_' + ','.join(file_names_test) + '_' + ','.join(str(temp) for temp in inputs_ft) + '_' + str(N_cycles_example) + ',' + str(N_cycles_step) + '_2g' + str(lstm_units) + '_' + str(lr)  # + '_f5,10,60'
save_filename = 'Ro={}_A={}_Tr={}_Te={}_in={}_bl={}_Nc={}_Ns={}_2L{}_lr={}'.format(
    Ro, A_star, 'all', ','.join(str(temp) for temp in sets_test),
    ','.join(str(temp) for temp in inputs_ft), baseline_d, N_cycles_example, N_cycles_step, lstm_units, lr)

# %% For 1 set at a time sets together
sets_to_plot = [1, 2, 3, 4, 5]
d_to_plot = [list(range(1, 43 + 1, 3))] * 5  # list of all distances from wall for each set
sets_to_plot_colors = ['rgba(255,0,0,1)', 'rgba(0,255,0,1)', 'rgba(0,0,255,1)', 'rgba(0,255,255,1)', 'rgba(255,255,0,1)']
sets_to_plot_colors_ci = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 'rgba(0,255,255,0.2)', 'rgba(255,255,0,0.2)']

layout = go.Layout(
    # width = 1280,
    # height = 720
)
figs = [go.Figure(layout=layout) for j in inputs_ft]

# %%
for set_to_plot_index in range(len(sets_to_plot)):
    sets_train = [sets_to_plot[set_to_plot_index]]
    d_train = [d_to_plot[set_to_plot_index]]
    d_train_labels = d_train

    # %%
    # test that the sets and distances are assigned correctly
    assert len(sets_train) == len(d_train)
    for i in range(len(sets_train)):
        assert len(d_train[i]) == len(d_train_labels[i])

    assert len(sets_test) == len(d_test)
    for i in range(len(sets_test)):
        assert len(d_test[i]) == len(d_test_labels[i])

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

    # data = (data - data_min) / (data_max - data_min)  # normalize
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

    # %% Find the average time-series for all examples at a distance
    d_all_labels = np.unique(file_labels)  # array of all distances to wall

    X_train_avg = np.zeros([len(d_all_labels), X_train.shape[1], N_inputs])
    X_train_std = np.zeros_like(X_train_avg)

    for d_index, d in enumerate(d_all_labels):
        X_train_d = X_train[y_train == d]
        X_train_avg[d_index] = np.mean(X_train_d, axis=0)
        X_train_std[d_index] = np.std(X_train_d, axis=0)

    # %% basic plt plot
    # # plt.plot(X_train_avg[0, :, 0])
    # plt.figure(figsize=(14, 8))
    # # sns.set_theme(style="white")
    # sns.lineplot(x=t[:N_per_example], y=X_train_avg[14, :, 0])
    # plt.fill_between(t[:N_per_example], (X_train_avg - 2 * X_train_std)[14, :, 0], (X_train_avg + 2 * X_train_std)[14, :, 0], color='b', alpha=.2)

    # %% Create figure
    for j in range(N_inputs):
        # layout = go.Layout(
        #     # width = 1280,
        #     # height = 720
        # )
        # fig = go.Figure(layout=layout)

        # # Add traces, one for each slider step
        # for step in np.arange(0, 5, 0.1):
        #     fig.add_trace(
        #         go.Scatter(
        #             visible=False,
        #             line=dict(color="#00CED1", width=1),
        #             mode='lines',
        #             # name="𝜈 = " + str(step),
        #             x=np.arange(0, 10, 1),
        #             y=np.sin(step * np.arange(0, 10, 1))))

        # # Make 10th trace visible
        # fig.data[10].visible = True

        # Add traces, one for each slider step
        for d_index in range(X_train_avg.shape[0]):
            figs[j].add_trace(
                go.Scatter(
                    visible=False,
                    # line=dict(color='rgba(255,0,0,1)', width=2),
                    line=dict(color=sets_to_plot_colors[set_to_plot_index], width=1),
                    mode='lines',
                    name="Set " + str(set_to_plot_index + 1),
                    x=t[:N_per_example],
                    y=X_train_avg[d_index, :, j]
                ),
            )

        for d_index in range(X_train_avg.shape[0]):
            figs[j].add_trace(
                go.Scatter(
                    visible=False,
                    mode='lines',
                    x=np.append(t[:N_per_example], t[:N_per_example][::-1]),  # x, then x reversed
                    y=np.append((X_train_avg[d_index, :, j] + 2 * X_train_std[d_index, :, j]), (X_train_avg[d_index, :, j] - 2 * X_train_std[d_index, :, j])[::-1]),  # upper, then lower reversed
                    fill='toself',
                    # fillcolor='rgba(255,0,0,0.2)',
                    fillcolor=sets_to_plot_colors_ci[set_to_plot_index],
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        # Make 1st distance trace visible
        figs[j].data[set_to_plot_index * len(d_all_labels) * 2].visible = True
        figs[j].data[len(d_all_labels) + set_to_plot_index * len(d_all_labels) * 2].visible = True

# %%
# Create and add slider
for j in range(N_inputs):
    steps = []
    for i in range(len(d_all_labels)):
        step = dict(
            method="update",
            # args=[{"visible": [False] * len(figs[j].data)//2},
            args=[{"visible": [False] * len(figs[j].data)},
                  {"title": "Distance (cm): " + str(d_all_labels[i])},  # layout attribute
                  ],
            label=str(d_all_labels[i])
        )
        for set_to_plot_index in range(len(sets_to_plot)):
            step["args"][0]["visible"][i + set_to_plot_index * len(d_all_labels) * 2] = True  # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i + len(d_all_labels) + set_to_plot_index * len(d_all_labels) * 2] = True  # Toggle i'th trace to "visible"

        steps.append(step)

    sliders = [dict(
        active=0,
        # currentvalue={"prefix": "Frequency: "},
        # pad={"t": 50},
        steps=steps
    )]

    figs[j].update_layout(
        sliders=sliders,
        template="simple_white",
        legend=dict(yanchor="top", xanchor="right")
    )

    figs[j].update_yaxes(range=(data_min[j], data_max[j]))

    figs[j].write_html(save_folder + save_filename + '/plot_input_' + str(j) + '.html')
    figs[j].show()

# %%
