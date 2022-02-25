# %%
import plotly.graph_objects as go
# import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns

# %%
# from helper_functions import divide_file_names, data_get_info, data_load, data_shorten_sequence
from helper_functions import data_full_process, y_norm_reverse
# from helper_functions import model_k_fold_tf
# from helper_functions import model_build_tf, model_fit_tf
# from helper_functions import model_predict_tf, model_evaluate_regression_tf
# import matplotlib.pyplot as plt
import numpy as np
# %load_ext autoreload
# %autoreload 2

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
Ro = 3.5
A_star = 2
Ro_d_last = {2: 46, 3.5: 43, 5: 40}  # furthest distance from wall for each wing shape

# all sets except the ones given in sets_val
# sets_train = [1, 2, 3, 4, 5]
# [sets_train.remove(set_val) for set_val in sets_val if set_val in sets_train]
# [sets_train.remove(set_test) for set_test in sets_test if set_test in sets_train]

sets_train = [1, 2, 3, 4, 5]
d_train = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_train)  # list of all distances from wall for each set
d_train_labels = d_train

sets_val = []
d_val = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_val)  # list of all distances from wall
d_val_labels = d_val

sets_test = []
d_test = [list(range(1, Ro_d_last[Ro] + 1, 3))] * len(sets_test)  # list of all distances from wall
d_test_labels = d_test

separate_val_files = len(sets_val) > 0
if separate_val_files:
    train_val_split = 1
    shuffle_examples = False
    shuffle_seed = None
else:
    train_val_split = 1
    shuffle_examples = False
    shuffle_seed = None
    # shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 14

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]
average_window = 10

baseline_d = 40  # set to None for no baseline

lstm_layers = 2
dense_hidden_layers = 1
N_units = 16  # number of lstm cells of each lstm layer
lr = 0.0002  # learning rate
dropout = 0.2
recurrent_dropout = 0.0
epochs_number = 10000  # number of epochs
epochs_patience = 10000  # for early stopping, set <0 to disable
# k_fold_splits = len(sets_train)

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2022.02.24_data_plot/'  # include trailing slash
save_filename = 'Ro={}_A={}_Tr={}_Val={}_Te={}_in={}_bl={}_Ne={}_Ns={}_win={}_{}L{}D{}_lr={}_dr={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
    ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft),
    baseline_d, N_cycles_example, N_cycles_step, average_window,
    lstm_layers, dense_hidden_layers, N_units, lr, dropout, recurrent_dropout)

# %% For 1 set at a time sets together
# sets_train = [1, 2, 3, 4, 5]
# d_train = [list(range(1, 43 + 1, 3))] * 5  # list of all distances from wall for each set
sets_train_colors = ['rgba(255,0,0,1)', 'rgba(0,255,0,1)', 'rgba(0,0,255,1)', 'rgba(0,255,255,1)', 'rgba(255,255,0,1)', 'rgba(255,0,255,1)']
sets_train_colors_ci = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 'rgba(0,255,255,0.2)', 'rgba(255,255,0,0.2)', 'rgba(255,0,255,0.2)']

layout = go.Layout(
    # width = 1280,
    # height = 720
)
figs = [go.Figure(layout=layout) for j in range(len(inputs_ft + inputs_ang))]

# %% load the data
X_train, y_train, s_train, X_val, y_val, s_val, X_min, X_max, y_min, y_max, X_baseline, N_per_example, N_inputs = \
    data_full_process(
        data_folder, Ro, A_star,
        sets_train, d_train, d_train_labels,
        sets_val, d_val, d_val_labels,
        inputs_ft, inputs_ang,
        N_cycles_example, N_cycles_step, N_cycles_to_use,
        train_val_split, separate_val_files, shuffle_examples, shuffle_seed,
        save_model, save_results, save_folder, save_filename,
        None, None, None, None, baseline_d, None, average_window
    )

# %% un-normalize y
y_train = np.round(y_norm_reverse(y_train, y_min, y_max))
y_val = np.round(y_norm_reverse(y_val, y_min, y_max))

# %% fix time length
# t = np.around(np.loadtxt(data_folder + file_names[0] + '/t.csv', delimiter=','), decimals=3)  # round to ms
# t_s = round(t[1] - t[0], 3)  # sample time
t_s = 0.005  # s
t_s = round(t_s * average_window, 3)  # sample time
t = np.arange(0, t_s * N_per_example, t_s)

# %% Find the average time-series for all examples at a distance
d_all_labels = np.unique(np.concatenate((y_train, y_val)))  # array of all distances to wall

for s_index in range(len(sets_train)):
    X_train_set = X_train[s_train == sets_train[s_index]]
    y_train_set = y_train[s_train == sets_train[s_index]]

    X_train_avg = np.zeros([len(d_train_labels[s_index]), X_train_set.shape[1], N_inputs])
    X_train_std = np.zeros_like(X_train_avg)

    for d_index, d in enumerate(d_train_labels[s_index]):
        X_train_d = X_train_set[y_train_set == d]
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
        # Add traces, one for each slider step
        for d_index in range(X_train_avg.shape[0]):
            figs[j].add_trace(
                go.Scatter(
                    visible=(d_index == 0),
                    # line=dict(color='rgba(255,0,0,1)', width=2),
                    line=dict(color=sets_train_colors[s_index], width=1),
                    mode='lines',
                    name="Set " + str(sets_train[s_index]),
                    x=t[:N_per_example],
                    y=X_train_avg[d_index, :, j]
                ),
            )

        for d_index in range(X_train_avg.shape[0]):
            figs[j].add_trace(
                go.Scatter(
                    visible=(d_index == 0),
                    mode='lines',
                    x=np.append(t[:N_per_example], t[:N_per_example][::-1]),  # x, then x reversed
                    y=np.append((X_train_avg[d_index, :, j] + 2 * X_train_std[d_index, :, j]), (X_train_avg[d_index, :, j] - 2 * X_train_std[d_index, :, j])[::-1]),  # upper, then lower reversed
                    fill='toself',
                    # fillcolor='rgba(255,0,0,0.2)',
                    fillcolor=sets_train_colors_ci[s_index],
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        # Make 1st distance trace visible
        # figs[j].data[s_index * len(d_labels_all) * 2].visible = True
        # figs[j].data[len(d_labels_all) + s_index * len(d_labels_all) * 2].visible = True

# %%
# Create and add slider
for j in range(N_inputs):
    steps = []
    for d_index, d in enumerate(d_all_labels):
        step = dict(
            method="update",
            # args=[{"visible": [False] * len(figs[j].data)//2},
            args=[{"visible": [False] * len(figs[j].data)},
                  {"title": "Distance (cm): " + str(d)},  # layout attribute
                  ],
            label=str(d)
        )

        d_fig_counter = 0
        for s_index in range(len(sets_train)):
            if d in d_train_labels[s_index]:
                d_index_in_set = d_train_labels[s_index].index(d)
                step["args"][0]["visible"][d_index_in_set + d_fig_counter] = True  # Toggle i'th trace to "visible"
                step["args"][0]["visible"][d_index_in_set + d_fig_counter + len(d_train_labels[s_index])] = True  # Toggle i'th trace to "visible"
            d_fig_counter += len(d_train_labels[s_index]) * 2

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

    figs[j].update_yaxes(range=(np.min(X_train[:, :, j]), np.max(X_train[:, :, j])))

    figs[j].write_html(save_folder + save_filename + '/plot_input_' + str(j) + '.html')
    figs[j].show()

# %%
