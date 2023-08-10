# %%
from helper_functions import data_full_process, y_norm_reverse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %load_ext autoreload
# %autoreload 2

# %% design parameters
root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
Ro = 5
A_star = 2

# bounds for distance from wall for each wing shape
Ro_d = {2: [1, 40], 3.5: [1, 40], 5: [1, 40]}
# Ro_d = {2: [10, 46], 3.5: [4, 40], 5: [1, 37]}  # same sensor-to-wall distance

sets_train = [1, 2, 3, 4, 5]
d_train = [list(range(Ro_d[Ro][0], Ro_d[Ro][1] + 1, 3))] * len(sets_train)  # list of all distances from wall for each set
d_train_labels = d_train

sets_val = []
d_val = [list(range(Ro_d[Ro][0], Ro_d[Ro][1] + 1, 3))] * len(sets_val)  # list of all distances from wall
d_val_labels = d_val

sets_test = []
d_test = [list(range(Ro_d[Ro][0], Ro_d[Ro][1] + 1, 3))] * len(sets_test)  # list of all distances from wall
d_test_labels = d_test

separate_val_files = len(sets_val) > 0
if separate_val_files:
    train_val_split = 1
    shuffle_seed = None
else:
    train_val_split = 1
    # shuffle_seed = np.random.default_rng().integers(0, high=1000)
    shuffle_seed = 5  # seed to split data in reproducible way

separate_test_files = len(sets_test) > 0
if separate_test_files:
    train_test_split = 1
    shuffle_seed = None
else:
    train_test_split = 1
    # shuffle_seed = np.random.default_rng().integers(0, high=1000)
    shuffle_seed = 5  # seed to split data in reproducible way

N_cycles_example = 1  # use this number of stroke cycles as 1 example
N_cycles_step = 1  # number of cycles to step between consecutive examples
# total number of cycles to use per file
# set 0 to automatically calculate number of examples from the first file
N_cycles_to_use = 14

inputs_ft = [0, 1, 2, 3, 4, 5]
inputs_ang = [0]

norm_X = False
norm_y = False
average_window = 5
baseline_d = None  # set to None for no baseline

save_model = True  # save model file, save last model if model_checkpoint == False
model_checkpoint = False  # doesn't do anything if save_model == False
save_results = True
save_folder = root_folder + 'plots/2023.08.04_forces/'  # include trailing slash
save_filename = 'Ro={}_A={}_Tr={}_Val={}_Te={}_inF={}_inA={}_bl={}_Ne={}_Ns={}_win={}_sh={}'.format(
    Ro, A_star, ','.join(str(temp) for temp in sets_train), ','.join(str(temp) for temp in sets_val),
    ','.join(str(temp) for temp in sets_test), ','.join(str(temp) for temp in inputs_ft), ','.join(str(temp) for temp in inputs_ang),
    baseline_d, N_cycles_example, N_cycles_step, average_window, shuffle_seed)

# %% load the data
# [X_mean, X_std, y_mean, y_std, X_baseline] = [None, None, None, None, None]  # initialize
X_mean = np.loadtxt(data_folder + 'Ro={}/A={}/X_mean.txt'.format(Ro, A_star))
X_std = np.loadtxt(data_folder + 'Ro={}/A={}/X_std.txt'.format(Ro, A_star))
y_mean = np.loadtxt(data_folder + 'Ro={}/A={}/y_mean.txt'.format(Ro, A_star))
y_std = np.loadtxt(data_folder + 'Ro={}/A={}/y_std.txt'.format(Ro, A_star))
X_baseline = None

X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,\
    X_mean, X_std, y_mean, y_std, X_baseline, N_per_example, N_inputs, t_s, t_cycle = \
    data_full_process(
        data_folder, Ro, A_star,
        sets_train, d_train, d_train_labels,
        sets_val, d_val, d_val_labels,
        sets_test, d_test, d_test_labels,
        inputs_ft, inputs_ang,
        N_cycles_example, N_cycles_step, N_cycles_to_use,
        separate_val_files, train_val_split, shuffle_seed,
        separate_test_files, train_test_split,
        save_model, save_folder, save_filename,
        norm_X, norm_y, X_mean, X_std, y_mean, y_std,
        baseline_d, X_baseline, average_window
    )

if norm_y:
    y_train = np.round(y_norm_reverse(y_train, y_mean, y_std))
    y_val = np.round(y_norm_reverse(y_val, y_mean, y_std))
    y_test = np.round(y_norm_reverse(y_test, y_mean, y_std))

t_s *= average_window

# %%
# plt.figure(0, figsize=[6, 4])

# d = 1

# m = np.mean(X_train[y_train == d, :, 0], axis=0)
# s = np.std(X_train[y_train == d, :, 0], axis=0)
# a = np.mean(X_train[y_train == d, :, 6], axis=0)
# t = np.arange(len(m)) * t_s

# plt.plot(t, m, label='{} cm'.format(d))
# plt.fill_between(t, m + s, m - s, alpha=0.3)

# d = 19

# m = np.mean(X_train[y_train == d, :, 0], axis=0)
# s = np.std(X_train[y_train == d, :, 0], axis=0)
# a = np.mean(X_train[y_train == d, :, 6], axis=0)
# t = np.arange(len(m)) * t_s

# plt.plot(t, m, label='{} cm'.format(d))
# plt.fill_between(t, m + s, m - s, alpha=0.3)

# d = 40

# m = np.mean(X_train[y_train == d, :, 0], axis=0)
# s = np.std(X_train[y_train == d, :, 0], axis=0)
# a = np.mean(X_train[y_train == d, :, 6], axis=0)
# t = np.arange(len(m)) * t_s

# plt.plot(t, m, label='{} cm'.format(d))
# plt.fill_between(t, m + s, m - s, alpha=0.3)

# plt.plot(t, a * 2 / 3, label='Phase')
# plt.ylim([-1, 1])
# plt.legend()
# plt.title('Ro={}, A*={}'.format(Ro, A_star))
# plt.xlabel('Time (s)')
# plt.ylabel('Normal Force')

# plt.savefig(save_folder + save_filename + '.svg')
# plt.show()
# plt.close()

# %% plot forces in small size without any borders (for neural network figure)
# fig, axs = plt.subplots(nrows=6, ncols=1, sharex=True, sharey=True, figsize=(1, 5))
# d = 1

# for i in range(6):
#     if i < 3:
#         fmt = 'r-'
#     else:
#         fmt = 'b-'

#     axs[i].plot(np.mean(X_train[y_train == d, :, i], axis=0), fmt)
#     axs[i].axis('off')

# plt.tight_layout()
# plt.savefig('forces_small.svg')

# # %%
# fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(1, 0.8))
# d = 1

# ax.plot(np.mean(X_train[y_train == d, :, 6], axis=0), 'g-')
# ax.axis('off')

# plt.tight_layout()
# plt.savefig('ang_small.svg')

# %% plot forces at all distances while changing transparency
# fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(6.5, 3.5))
# d_all = np.unique(y_train)
# trans = [0.2, 1]

# for i in range(2):
#     for j in range(3):
#         for d in d_all:
#             if i * 3 + j < 3:
#                 fmt = 'r-'
#             else:
#                 fmt = 'b-'

#             alpha = (d - d_all.min()) / (d_all.max() - d_all.min()) * (trans[1] - trans[0]) + trans[0]
#             axs[i, j].plot(np.mean(X_train[y_train == d, :, i * 3 + j], axis=0), fmt, alpha=alpha, linewidth=0.3)
#             # axs[i].axis('off')

# plt.tight_layout()
# plt.savefig(save_folder + save_filename + '.svg')

# %% plot forces at all distances while changing color spectrum
# Set default font size and type
plt.rcParams['font.size'] = '10'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    "savefig.facecolor": (1, 1, 1, 1),  # disable transparent background
    "axes.titlesize": 10,
})
# plt.rc('font', family='serif', size=10)

horiz = False  # 2x3 or 3x2 layout?
if horiz:
    nrows, ncols = [2, 3]
    figsize = (6, 3.5)
else:
    nrows, ncols = [3, 2]
    figsize = (5, 5.5)

fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=figsize,
    # constrained_layout=True,
    # cbar_location="right",
    # cbar_mode="single",
    # cbar_size="7%",
    # cbar_pad=0.15
)
d_all = np.unique(y_train)
d_all = np.flip(d_all)  # to plot further distances first, for changing the overlapping
gradient = np.linspace(0, 1, len(d_all))
cmap = cm.get_cmap('viridis')
ylabels = ['Normal Force (N)', 'Spanwise Force (N)', 'Chordwise Force (N)', 'Normal Torque (N-mm)', 'Spanwise Torque (N-mm)', 'Chordwise Torque (N-mm)']
ylims = [[-0.4, 0.4], [-0.025, -0.005], [-0.02, 0.02], [-2.2, 2.2], [-0.65, 0.65], [-60, 60]]
yticks = [None, [-0.01, -0.02], None, None, [-0.6, -0.3, 0, 0.3, 0.6], None]
t = np.arange(X_train.shape[1]) * t_s
phase = np.arange(X_train.shape[1]) / X_train.shape[1] * 360

for i in range(nrows):
    for j in range(ncols):
        for d_i, d in enumerate(d_all):
            if horiz:
                n = i * ncols + j
            else:
                n = i + j * nrows

            axs[i, j].plot(phase, np.mean(X_train[y_train == d, :, n], axis=0), color=cmap(gradient[d_i]), linewidth=0.5)
            # axs[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[i, j].set_title(ylabels[n], y=1)
            axs[i, j].set_ylim(ylims[n])
            # axs[i, j].grid(True)

            # mark stroke reversal
            axs[i, j].axvline(90, ymin=0, ymax=1, color='black', linestyle=':', linewidth=1)
            axs[i, j].axvline(270, ymin=0, ymax=1, color='black', linestyle=':', linewidth=1)

            if yticks[n] is not None:
                axs[i, j].set_yticks(yticks[n])

axs[0, 0].set_xlim([phase[0], phase[-1]])
axs[0, 0].set_xticks([0, 90, 180, 270, 360])
fig.supxlabel(r'Phase ($\degree$)', fontsize=10)
fig.tight_layout()

# add color bar
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.89, 0.124, 0.025, 0.817])  # xloc, yloc, width, height
# cax.autoscale(tight=True)
make_axes_locatable(cax)
mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=gradient, orientation="vertical")
# cax.set_yticks([0, 1])
# cax.set_yticklabels(['{:.0f}'.format(d_all[d_i]) for d_i in [0, -1]])
cax.set_yticks(gradient)
cax.set_yticklabels(np.round(d_all).astype(int))
cax.invert_yaxis()
cax.set_ylabel(r'Plate Tip-to-Wall Distance ($d_{tip}$, cm)')  # , rotation=270, va='bottom')
cax.tick_params(size=0)

# save figure
# plt.savefig(save_folder + save_filename + '.eps')
plt.savefig(save_folder + 'forces.eps')
plt.savefig(save_folder + 'forces.png', dpi=300)

# %%
