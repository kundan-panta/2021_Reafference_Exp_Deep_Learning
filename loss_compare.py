# %%
import numpy as np
from os import walk
import matplotlib.pyplot as plt
from pathlib import Path

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]

# initialize NaN array to hold the losses for all cases
losses_mean = np.full((len(Ro_all), len(A_star_all)), np.NaN)
losses_std = np.full((len(Ro_all), len(A_star_all)), np.NaN)

# assume there's only 1 folder for each case
root_folder = ''  # include trailing slash
# save_folders = ['plots/2022.03.25_exp_best_sh=5/', 'plots/2022.03.25_exp_best_sh=50/', 'plots/2022.03.25_exp_best_sh=500/', 'plots/2022.04.14_exp_sh=5000/']  # include trailing slash
save_folders = ['plots/2022.03.26_exp_best_same_sensor_loc_sh=500/']  # include trailing slash
save_folders = [root_folder + save_folder for save_folder in save_folders]
plot_folder = root_folder + 'plots/2022.04.14_loss_plots/sensor-to-wall/'

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
# ylim = [0, 8]
# plt.tight_layout()

# fig = plt.figure(figsize=[16, 4])
# fig.supxlabel('Ro')
# fig.supylabel('Mean Absolute Error (cm)')

# for A_star_ind, A_star in enumerate(A_star_all):
#     losses_boxplot = []
#     for Ro_ind, Ro in enumerate(Ro_all):
#         losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#     plt.subplot(1, 3, A_star_ind + 1)
#     plt.boxplot(losses_boxplot, showfliers=False)
#     plt.xticks([1, 2, 3], Ro_all)
#     plt.title('A* = {}'.format(A_star))
#     # plt.ylim(ylim)

# fig = plt.figure(figsize=[16, 4])
# fig.supxlabel('A*')
# fig.supylabel('Mean Absolute Error (cm)')

# for Ro_ind, Ro in enumerate(Ro_all):
#     losses_boxplot = []
#     for A_star_ind, A_star in enumerate(A_star_all):
#         losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

#     plt.subplot(1, 3, Ro_ind + 1)
#     plt.boxplot(losses_boxplot, showfliers=False)
#     plt.xticks([1, 2, 3], A_star_all)
#     plt.title('Ro = {}'.format(Ro))
#     # plt.ylim(ylim)

# %% make loss plots with multiple rows for each distance
# create folder to save plots to
Path(plot_folder).mkdir(parents=True, exist_ok=True)

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

# with distance on x-axis
d_all = [list(range(1, 40 + 1, 3))] * len(Ro_all)
# d_all = [list(range(10, 46 + 1, 3)), list(range(4, 40 + 1, 3)), list(range(1, 37 + 1, 3))]  # if using same sensor-to-wall distance

# find distance from opposite wall of tank
wing_len_Ro = [12.367, 17.059, 20.827]
tank_len = 81  # tank length (cm)
d_all_opp = []
for Ro_ind in range(len(d_all)):
    d_all_opp.append([tank_len - (wing_len_Ro[Ro_ind] + d) for d in d_all[Ro_ind]])

# put the Ro's close together
figs_A_star = [0] * len(A_star_all)
axs_A_star = [0] * len(A_star_all)

for i in range(len(A_star_all)):
    figs_A_star[i], axs_A_star[i] = plt.subplots(len(Ro_all), 1, sharex=True, sharey=True, figsize=(15, 10))

for A_star_ind, A_star in enumerate(A_star_all):
    figs_A_star[A_star_ind].suptitle('A* = {}'.format(A_star))
    figs_A_star[A_star_ind].supxlabel('Distance (cm)')
    figs_A_star[A_star_ind].supylabel('MAE (cm)')

    for Ro_ind, Ro in enumerate(Ro_all):
        losses_boxplot = []
        for d_ind, d in enumerate(d_all[Ro_ind]):  # one boxplot for each distance
            losses_boxplot.append(load_loss(Ro, A_star, d, d))

        axs_A_star[A_star_ind][Ro_ind].boxplot(losses_boxplot, showfliers=False)
        plt.setp(axs_A_star[A_star_ind][Ro_ind], xticks=list(range(1, len(d_all[Ro_ind]) + 1)), xticklabels=d_all[Ro_ind])
        axs_A_star[A_star_ind][Ro_ind].set_ylim(0, 10)
        axs_A_star[A_star_ind][Ro_ind].set_ylabel('Ro = {}'.format(Ro))

        # line going through median
        # axs_A_star[A_star_ind][Ro_ind].plot(list(range(1, len(d_all[Ro_ind]) + 1)), np.median(losses_boxplot, axis=1), 'orange')

    # save
    figs_A_star[A_star_ind].savefig(plot_folder + 'A={}.png'.format(A_star))


# put the A*'s close together
figs_Ro = [0] * len(Ro_all)
axs_Ro = [0] * len(Ro_all)
ylim_Ro = [2, 10, 10]


for i in range(len(Ro_all)):
    figs_Ro[i], axs_Ro[i] = plt.subplots(len(Ro_all), 1, sharex=True, sharey=True, figsize=(15, 10))

for Ro_ind, Ro in enumerate(Ro_all):
    figs_Ro[Ro_ind].suptitle('Ro = {}'.format(Ro))
    figs_Ro[Ro_ind].supxlabel('Distance (cm)')
    figs_Ro[Ro_ind].supylabel('MAE (cm)')

    for A_star_ind, A_star in enumerate(A_star_all):
        losses_boxplot = []
        for d_ind, d in enumerate(d_all[Ro_ind]):  # one boxplot for each distance
            losses_boxplot.append(load_loss(Ro, A_star, d, d))

        axs_Ro[Ro_ind][A_star_ind].boxplot(losses_boxplot, showfliers=False)
        plt.setp(axs_Ro[Ro_ind][A_star_ind], xticks=list(range(1, len(d_all[Ro_ind]) + 1)), xticklabels=d_all[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_ylim(0, ylim_Ro[Ro_ind])
        axs_Ro[Ro_ind][A_star_ind].set_ylabel('A* = {}'.format(A_star))

        # line going through median
        # axs_Ro[Ro_ind][A_star_ind].plot(list(range(1, len(d_all[Ro_ind]) + 1)), np.median(losses_boxplot, axis=1), 'orange')

    # save
    figs_Ro[Ro_ind].savefig(plot_folder + 'Ro={}.png'.format(Ro))

# %%
plt.show()
plt.close()

# %%
