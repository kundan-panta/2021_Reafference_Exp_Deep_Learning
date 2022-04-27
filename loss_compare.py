# %%
import numpy as np
from os import walk
import matplotlib.pyplot as plt
from pathlib import Path

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]

# initialize NaN array to hold the losses for all cases
# losses_mean = np.full((len(Ro_all), len(A_star_all)), np.NaN)
# losses_std = np.full((len(Ro_all), len(A_star_all)), np.NaN)

root_folder = ''  # include trailing slash
plot_folder = root_folder + 'plots/2022.04.27_loss_plots/'

# choose between wingtip-to-wall or sensor-to-wall models
w2w = True
if w2w:
    save_folders = [
        'plots/2022.03.25_exp_best_sh=5/',
        'plots/2022.03.25_exp_best_sh=50/',
        'plots/2022.03.25_exp_best_sh=500/',
        'plots/2022.04.14_exp_sh=5000/',
        'plots/2022.04.16_exp_sh=50000/'
    ]  # include trailing slash
    plot_folder += 'w2w/'
else:
    save_folders = [
        'plots/2022.03.26_exp_best_same_sensor_loc_sh=500/',
        'plots/2022.04.17_exp_best_same_sensor_loc_sh=5/',
        'plots/2022.04.19_exp_best_same_sensor_loc_sh=50/',
        'plots/2022.04.19_exp_best_same_sensor_loc_sh=5000/',
        'plots/2022.04.25_exp_best_same_sensor_loc_sh=50000/'
    ]  # include trailing slash
    plot_folder += 's2w/'

save_folders = [root_folder + save_folder for save_folder in save_folders]

# %% mean and std plots


def load_loss(Ro, A_star, d_min, d_max):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)

    losses_case_all_folders = []
    for save_folder in save_folders:
        _, folders, _ = next(walk(save_folder), (None, [], None))
        for folder in folders:
            if target_string in folder:
                losses_case = np.loadtxt(save_folder + folder + '/loss_test_all.txt')
                y_case = np.loadtxt(save_folder + folder + '/y_test.txt')
                # choose only losses within the desired distance
                losses_case_all_folders.append(losses_case[np.logical_and(y_case >= d_min, y_case <= d_max)])
    return np.concatenate(losses_case_all_folders)


def loss_mean_std(losses_case):
    return np.mean(losses_case), np.std(losses_case)


# # distances between which to get losses
# d_min = 0
# d_max = 99

# for Ro_ind, Ro in enumerate(Ro_all):
#     for A_star_ind, A_star in enumerate(A_star_all):
#         losses_mean[Ro_ind, A_star_ind], losses_std[Ro_ind, A_star_ind] = loss_mean_std(load_loss(Ro, A_star, d_min, d_max))

# # make figures
# line_types = ['ro--', 'g*-.', 'b^:']

# fig_Ro_loss = plt.figure()
# for A_star_ind, A_star in enumerate(A_star_all):
#     plt.errorbar(Ro_all, losses_mean[:, A_star_ind], yerr=losses_std[:, A_star_ind], label='A*={}'.format(A_star), fmt=line_types[A_star_ind], capsize=5)
# plt.legend()
# plt.xlabel('Ro')
# plt.ylabel('Test Set Loss')

# fig_A_star_loss = plt.figure()
# for Ro_ind, Ro in enumerate(Ro_all):
#     plt.errorbar(A_star_all, losses_mean[Ro_ind, :], yerr=losses_std[Ro_ind, :], label='Ro={}'.format(Ro), fmt=line_types[Ro_ind], capsize=5)
# plt.legend()
# plt.xlabel('A*')
# plt.ylabel('Test Set Loss')

# %% boxplot
# # distances between which to get losses
# d_min = 0
# d_max = 99

# ylim = [0, 8]
# plt.tight_layout()

# fig = plt.figure(figsize=[16, 4])
# fig.supxlabel('Ro')
# fig.supylabel('|Prediction Error| (cm)')

# for A_star_ind, A_star in enumerate(A_star_all):
#     losses_boxplot = []
#     for Ro_ind, Ro in enumerate(Ro_all):
#         losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#     plt.subplot(1, 3, A_star_ind + 1)
#     plt.boxplot(losses_boxplot, showfliers=False, notch=True)
#     plt.xticks([1, 2, 3], Ro_all)
#     plt.title('A* = {}'.format(A_star))
#     plt.ylim(ylim)

# fig = plt.figure(figsize=[16, 4])
# fig.supxlabel('A*')
# fig.supylabel('|Prediction Error| (cm)')

# for Ro_ind, Ro in enumerate(Ro_all):
#     losses_boxplot = []
#     for A_star_ind, A_star in enumerate(A_star_all):
#         losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#     plt.subplot(1, 3, Ro_ind + 1)
#     plt.boxplot(losses_boxplot, showfliers=False, notch=True)
#     plt.xticks([1, 2, 3], A_star_all)
#     plt.title('Ro = {}'.format(Ro))
#     plt.ylim(ylim)

# %% prettier box plots
# distances between which to get losses
d_min = 0
d_max = 99

ylim = [-0.02, 8]

fig, ax = plt.subplots(figsize=[6.5, 4])
# fig.supxlabel('Ro')
# fig.supylabel('|Prediction Error| (cm)')

losses_boxplot_A = []
for A_star_ind, A_star in enumerate(A_star_all):
    losses_boxplot = []
    for Ro_ind, Ro in enumerate(Ro_all):
        losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))
    losses_boxplot_A.append(losses_boxplot)

# --- Labels for your data:
labels_list = ['2', '3.5', '5']
width = 0.3
xlocations = [x * ((1 + len(losses_boxplot_A)) * width) for x in range(len(losses_boxplot_A[0]))]

symbol = 'r+'
# ymin = min([val for dg in losses_boxplot_A for data in dg for val in data])
# ymax = max([val for dg in losses_boxplot_A for data in dg for val in data])

# ax = plt.gca()
# ax.set_ylim(ymin, ymax)
ax.set_ylim(ylim[0], ylim[1])

ax.grid(True, linestyle=':', axis='y')
ax.set_axisbelow(True)

plt.xlabel('Ro')
plt.ylabel('|Prediction Error| (cm)')
# plt.title('title')

space = len(losses_boxplot_A) / 2
offset = len(losses_boxplot_A) / 2


# --- Offset the positions per group:

group_positions = []
for num, dg in enumerate(losses_boxplot_A):
    _off = (0 - space + (0.5 + num))
    print(_off)
    group_positions.append([x + _off * (width + 0.01) for x in xlocations])

boxplots = [0] * len(A_star_all)
colors_bp = ['pink', 'lightblue', 'lightgreen']
for i, (dg, pos) in enumerate(zip(losses_boxplot_A, group_positions)):
    boxplots[i] = ax.boxplot(dg,
                             sym=symbol,
                             labels=[''] * len(labels_list),
                             positions=pos,
                             widths=width,
                             notch=False,
                             #   vert=True,
                             #   whis=1.5,
                             #   bootstrap=None,
                             #   usermedians=None,
                             #   conf_intervals=None,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors_bp[i]),
                             showfliers=False
                             )

# color the boxplots


def color_box(bp, color):
    # Define the elements to color. You can also add medians, fliers and means
    elements = ['boxes']
    elements = []

    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color) for idx in range(len(bp[elem]))]

    # Common colors for all boxplots
    [plt.setp(bp['medians'][idx], color='black') for idx in range(len(bp['medians']))]


for i, bp in enumerate(boxplots):
    color_box(bp, colors_bp[i])

# formatting stuff
ax.set_xticks(xlocations)
ax.set_xticklabels(labels_list, rotation=0)
ax.tick_params(axis='both', width=0)
plt.box(False)

# add legend
labels_legend = ['A*=2', 'A*=3', 'A*=4']
ax.legend([bp["boxes"][0] for bp in boxplots], labels_legend, loc='best')

plt.tight_layout()
Path(plot_folder).mkdir(parents=True, exist_ok=True)
plt.savefig(plot_folder + 'summary.svg')
plt.show()

# %% make loss plots with multiple rows for each distance
# d_rows = [[1, 1], [4, 4], [7, 7], [10, 10], [13, 13], [16, 16], [19, 19], [22, 22], [25, 25], [28, 28], [31, 31], [34, 34], [37, 37], [40, 40]]
# d_rows = [[1, 4], [7, 10], [13, 16], [19, 22], [25, 28], [31, 34], [37, 40]]
# d_rows = [[1, 7], [10, 16], [19, 25], [28, 34], [37, 40]]
# d_rows = [[1, 10], [13, 22], [25, 34], [37, 40]]
# d_rows = [[1, 13], [16, 28], [31, 40]]
# n_rows = len(d_rows)

# wrt Ro
# fig, axs = plt.subplots(n_rows, len(A_star_all), sharex=True, sharey=True, figsize=(12, 10))
# # fig.supxlabel('Ro')
# # fig.supylabel('Mean Absolute Error (cm)')

# for i_row, d_row in enumerate(d_rows):
#     d_min, d_max = d_row

#     for A_star_ind, A_star in enumerate(A_star_all):
#         losses_boxplot = []
#         for Ro_ind, Ro in enumerate(Ro_all):  # one box for each Ro in the same axes
#             losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#         axs[i_row, A_star_ind].boxplot(losses_boxplot, showfliers=False)
#         # axs[i_row, A_star_ind].set_xticks([1, 2, 3], Ro_all)
#         # axs[i_row, A_star_ind].set_title('A* = {}'.format(A_star))
#         plt.setp(axs[i_row, A_star_ind], xticks=[1, 2, 3], xticklabels=Ro_all)

# plt.ylim([0, 10])

# wrt A*
# fig, axs = plt.subplots(n_rows, len(A_star_all), sharex=True, sharey=True, figsize=(12, 13))
# # fig.supxlabel('Ro')
# # fig.supylabel('Mean Absolute Error (cm)')

# for i_row, d_row in enumerate(d_rows):
#     d_min, d_max = d_row

#     for Ro_ind, Ro in enumerate(Ro_all):
#         losses_boxplot = []
#         for A_star_ind, A_star in enumerate(A_star_all):  # one box for each A* in the same axes
#             losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#         axs[i_row, Ro_ind].boxplot(losses_boxplot, showfliers=False)
#         plt.setp(axs[i_row, Ro_ind], xticks=[1, 2, 3], xticklabels=A_star_all)

# plt.ylim([0, 10])

# %% with distance on x-axis
# create folder to save plots to
Path(plot_folder).mkdir(parents=True, exist_ok=True)

if w2w:
    d_all = [list(range(1, 40 + 1, 3))] * len(Ro_all)
else:
    d_all = [list(range(10, 46 + 1, 3)), list(range(4, 40 + 1, 3)), list(range(1, 37 + 1, 3))]

# find wingroot-to-wall distance from opposite wall of tank
wing_len_Ro = [12.367, 17.059, 20.827]
tank_len = 81  # tank length (cm)
d_all_opp = []
for Ro_ind in range(len(d_all)):
    d_all_opp.append([round(tank_len - (wing_len_Ro[Ro_ind] + d), 1) for d in d_all[Ro_ind]])

# use wingroot-to-wall instead of wingtip-to-wall in the plot
sensor_to_wall_distance = False

if sensor_to_wall_distance:
    # calculate the wingroot-to-wall distance
    d_all_label = []
    for Ro_ind in range(len(d_all)):
        d_all_label.append([round(wing_len_Ro[Ro_ind] + d, 1) for d in d_all[Ro_ind]])
    # don't share the x-axis when comparing different Ro b/c the wingroot-to-wall distances will be different
    sharex = False
else:
    d_all_label = d_all
    sharex = True

# put the Ro's close together
figs_A_star = [0] * len(A_star_all)
axs_A_star = [0] * len(A_star_all)

# for opposite wall distance
axs_opp_Ro = [[0] * len(Ro_all)] * len(A_star_all)
color_opp = 'green'

for i in range(len(A_star_all)):
    figs_A_star[i], axs_A_star[i] = plt.subplots(len(Ro_all), 1, sharex=sharex, sharey=True, figsize=(15, 10))

for A_star_ind, A_star in enumerate(A_star_all):
    figs_A_star[A_star_ind].suptitle('A* = {}'.format(A_star))
    figs_A_star[A_star_ind].supxlabel('Distance (cm)')
    figs_A_star[A_star_ind].supylabel('|Prediction Error| (cm)')

    for Ro_ind, Ro in enumerate(Ro_all):
        losses_boxplot = []
        for d_ind, d in enumerate(d_all[Ro_ind]):  # one boxplot for each distance
            losses_boxplot.append(load_loss(Ro, A_star, d, d))

        axs_A_star[A_star_ind][Ro_ind].boxplot(losses_boxplot, showfliers=False, notch=True)
        plt.setp(axs_A_star[A_star_ind][Ro_ind], xticks=list(range(1, len(d_all[Ro_ind]) + 1)), xticklabels=d_all_label[Ro_ind])
        axs_A_star[A_star_ind][Ro_ind].set_ylim(0, 10)
        axs_A_star[A_star_ind][Ro_ind].set_ylabel('Ro = {}'.format(Ro))

        # line going through median
        axs_A_star[A_star_ind][Ro_ind].plot(list(range(1, len(d_all[Ro_ind]) + 1)), np.median(losses_boxplot, axis=1), 'orange')

        # show distance to opposite wall ticks
        axs_opp_Ro[A_star_ind][Ro_ind] = axs_A_star[A_star_ind][Ro_ind].secondary_xaxis('top')
        plt.setp(axs_opp_Ro[A_star_ind][Ro_ind], xticks=list(range(1, len(d_all_opp[Ro_ind]) + 1)), xticklabels=d_all_opp[Ro_ind])
        axs_opp_Ro[A_star_ind][Ro_ind].tick_params(axis='x', colors=color_opp)

    # hide x-axis labels from the subplots below the first one for opposite wall distance
    # for Ro_ind in range(1, len(Ro_all)):
    #     axs_opp_Ro[A_star_ind][Ro_ind].set_xticklabels([])

    axs_opp_Ro[A_star_ind][0].set_xlabel('Distance from opposite wall (cm)', color=color_opp)

    # save
    figs_A_star[A_star_ind].savefig(plot_folder + 'A={}.png'.format(A_star))


# put the A*'s close together
figs_Ro = [0] * len(Ro_all)
axs_Ro = [0] * len(Ro_all)
ylim_Ro = [16, 16, 16]

# for opposite wall distance
axs_opp_Ro = [[0] * len(Ro_all)] * len(A_star_all)
color_opp = 'green'

for i in range(len(Ro_all)):
    figs_Ro[i], axs_Ro[i] = plt.subplots(len(Ro_all), 1, sharex=True, sharey=True, figsize=(15, 10))

for Ro_ind, Ro in enumerate(Ro_all):
    figs_Ro[Ro_ind].suptitle('Ro = {}'.format(Ro))
    figs_Ro[Ro_ind].supxlabel('Distance (cm)')
    figs_Ro[Ro_ind].supylabel('|Prediction Error| (cm)')

    for A_star_ind, A_star in enumerate(A_star_all):
        losses_boxplot = []
        for d_ind, d in enumerate(d_all[Ro_ind]):  # one boxplot for each distance
            losses_boxplot.append(load_loss(Ro, A_star, d, d))

        axs_Ro[Ro_ind][A_star_ind].boxplot(losses_boxplot, showfliers=False, notch=True)
        plt.setp(axs_Ro[Ro_ind][A_star_ind], xticks=list(range(1, len(d_all[Ro_ind]) + 1)), xticklabels=d_all_label[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_ylim(0, ylim_Ro[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_ylabel('A* = {}'.format(A_star))

        # line going through median
        axs_Ro[Ro_ind][A_star_ind].plot(list(range(1, len(d_all[Ro_ind]) + 1)), np.median(losses_boxplot, axis=1), 'orange')

        # show distance to opposite wall ticks
        axs_opp_Ro[Ro_ind][A_star_ind] = axs_Ro[Ro_ind][A_star_ind].secondary_xaxis('top')
        plt.setp(axs_opp_Ro[Ro_ind][A_star_ind], xticks=list(range(1, len(d_all_opp[Ro_ind]) + 1)), xticklabels=d_all_opp[Ro_ind])
        axs_opp_Ro[Ro_ind][A_star_ind].tick_params(axis='x', colors=color_opp)

    # hide x-axis labels from the subplots below the first one for opposite wall distance
    for A_star_ind in range(1, len(A_star_all)):
        axs_opp_Ro[Ro_ind][A_star_ind].set_xticklabels([])

    axs_opp_Ro[Ro_ind][0].set_xlabel('Distance from opposite wall (cm)', color=color_opp)

    # save
    figs_Ro[Ro_ind].savefig(plot_folder + 'Ro={}.png'.format(Ro))

# %% with distance, but prettier, only lines, no boxplots
# create folder to save plots to
Path(plot_folder).mkdir(parents=True, exist_ok=True)

if w2w:
    d_all = [list(range(1, 40 + 1, 3))] * len(Ro_all)
else:
    d_all = [list(range(10, 46 + 1, 3)), list(range(4, 40 + 1, 3)), list(range(1, 37 + 1, 3))]

# find wingroot-to-wall distance from opposite wall of tank
wing_len_Ro = [12.367, 17.059, 20.827]
tank_len = 81  # tank length (cm)
d_all_opp = []
for Ro_ind in range(len(d_all)):
    d_all_opp.append([round(tank_len - (wing_len_Ro[Ro_ind] + d), 1) for d in d_all[Ro_ind]])

# use wingroot-to-wall instead of wingtip-to-wall in the plot
sensor_to_wall_distance = False

if sensor_to_wall_distance:
    # calculate the wingroot-to-wall distance
    d_all_label = []
    for Ro_ind in range(len(d_all)):
        d_all_label.append([round(wing_len_Ro[Ro_ind] + d, 1) for d in d_all[Ro_ind]])
    # don't share the x-axis when comparing different Ro b/c the wingroot-to-wall distances will be different
    sharex = False
else:
    d_all_label = d_all
    sharex = True

# put the A*'s close together
figs_Ro = [0] * len(Ro_all)
axs_Ro = [0] * len(Ro_all)
ylim_Ro = [[0.00001, 16], [0.00001, 16], [0.00001, 16]]
# ylim_Ro = [[-3, 1.2], [-3, 1.2], [-3, 1.2]]
xlim_Ro = [81, 81, 81]

# for opposite wall distance
axs_opp_Ro = [[0] * len(Ro_all)] * len(A_star_all)
color_opp = 'green'

for i in range(len(Ro_all)):
    figs_Ro[i], axs_Ro[i] = plt.subplots(len(Ro_all), 1, sharex=True, sharey=True, figsize=(7.5, 5))

for Ro_ind, Ro in enumerate(Ro_all):
    figs_Ro[Ro_ind].suptitle('Ro = {}'.format(Ro))
    figs_Ro[Ro_ind].supxlabel('Distance (cm)')
    figs_Ro[Ro_ind].supylabel('|Prediction Error| (cm)')

    for A_star_ind, A_star in enumerate(A_star_all):
        losses_case = []
        for d_ind, d in enumerate(d_all[Ro_ind]):  # one boxplot for each distance
            losses_case.append(load_loss(Ro, A_star, d, d))

        # make boxplot
        # axs_Ro[Ro_ind][A_star_ind].boxplot(losses_case, showfliers=False, notch=True)
        # line going through median
        # axs_Ro[Ro_ind][A_star_ind].plot(list(range(1, len(d_all[Ro_ind]) + 1)), np.median(losses_case, axis=1), 'orange')

        # axis formatting
        # plt.setp(axs_Ro[Ro_ind][A_star_ind], xticks=list(range(1, len(d_all[Ro_ind]) + 1)), xticklabels=d_all_label[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_yscale('log')
        axs_Ro[Ro_ind][A_star_ind].set_ylim(ylim_Ro[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_ylabel('A* = {}'.format(A_star))
        axs_Ro[Ro_ind][A_star_ind].set_xlim(0, xlim_Ro[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].tick_params(left=False, bottom=False)
        # gridlines
        axs_Ro[Ro_ind][A_star_ind].grid(True, linestyle=':', axis='both')
        # hide frame
        axs_Ro[Ro_ind][A_star_ind].spines['top'].set_visible(False)
        axs_Ro[Ro_ind][A_star_ind].spines['right'].set_visible(False)
        # axs_Ro[Ro_ind][A_star_ind].spines['bottom'].set_visible(False)
        # axs_Ro[Ro_ind][A_star_ind].spines['left'].set_visible(False)

        # make lineplots instead of boxplots
        # get the whisker values of boxplots
        iqr = np.percentile(losses_case, 75, axis=1) - np.percentile(losses_case, 25, axis=1)
        upper = np.percentile(losses_case, 75, axis=1) + 1.5 * iqr
        upper = np.minimum(upper, np.max(losses_case, axis=1))
        lower = np.percentile(losses_case, 25, axis=1) - 1.5 * iqr
        lower = np.maximum(lower, np.min(losses_case, axis=1))
        # make the lineplots
        axs_Ro[Ro_ind][A_star_ind].plot(d_all_label[Ro_ind], np.median(losses_case, axis=1), color='red', linestyle='-', marker='o')
        axs_Ro[Ro_ind][A_star_ind].plot(d_all_label[Ro_ind], upper, color='blue', linestyle='-', marker='+', linewidth=0.5)
        axs_Ro[Ro_ind][A_star_ind].plot(d_all_label[Ro_ind], lower, color='blue', linestyle='-', marker='+', linewidth=0.5)

        # show distance to opposite wall ticks
        # axs_opp_Ro[Ro_ind][A_star_ind] = axs_Ro[Ro_ind][A_star_ind].secondary_xaxis('top')
        # plt.setp(axs_opp_Ro[Ro_ind][A_star_ind], xticks=list(range(1, len(d_all_opp[Ro_ind]) + 1)), xticklabels=d_all_opp[Ro_ind])
        # axs_opp_Ro[Ro_ind][A_star_ind].tick_params(axis='x', colors=color_opp)

    # hide x-axis labels from the subplots below the first one for opposite wall distance
    # for A_star_ind in range(1, len(A_star_all)):
    #     axs_opp_Ro[Ro_ind][A_star_ind].set_xticklabels([])
    # axs_opp_Ro[Ro_ind][0].set_xlabel('Distance from opposite wall (cm)', color=color_opp)

    # save
    figs_Ro[Ro_ind].tight_layout()
    figs_Ro[Ro_ind].savefig(plot_folder + 'Ro={}.png'.format(Ro))

# %% with distance, boxplots, but pretty

# %%
plt.show()
plt.close()

# %%
