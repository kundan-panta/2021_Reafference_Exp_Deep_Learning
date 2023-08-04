# %%
import numpy as np
from os import walk
import matplotlib.pyplot as plt
# from matplotlib import cm
from pathlib import Path

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]

# initialize NaN array to hold the losses for all cases
# losses_mean = np.full((len(Ro_all), len(A_star_all)), np.NaN)
# losses_std = np.full((len(Ro_all), len(A_star_all)), np.NaN)

root_folder = ''  # include trailing slash
plot_folder = root_folder + 'plots/2022.05.17_predictions_noshuffle/'

# choose between wingtip-to-wall or sensor-to-wall models
w2w = True
if w2w:
    # save_folders = [
    #     'plots/2022.03.25_exp_best_sh=5/',
    #     'plots/2022.03.25_exp_best_sh=50/',
    #     'plots/2022.03.25_exp_best_sh=500/',
    #     'plots/2022.04.14_exp_best_sh=5000/',
    #     'plots/2022.04.16_exp_best_sh=50000/'
    # ]  # include trailing slash
    save_folders = [
        'plots/2022.05.03_noshuffle_Te=1_sh=5/',
        'plots/2022.05.07_noshuffle_Te=2_sh=5/',
        'plots/2022.05.07_noshuffle_Te=3_sh=5/',
        'plots/2022.05.07_noshuffle_Te=4_sh=5/',
        'plots/2022.05.10_noshuffle_Te=5_sh=5/'
    ]  # include trailing slash
    suffix = 'w2w'

else:
    save_folders = [
        'plots/2022.03.26_exp_best_same_sensor_loc_sh=500/',
        'plots/2022.04.17_exp_best_same_sensor_loc_sh=5/',
        'plots/2022.04.19_exp_best_same_sensor_loc_sh=50/',
        'plots/2022.04.19_exp_best_same_sensor_loc_sh=5000/',
        'plots/2022.04.25_exp_best_same_sensor_loc_sh=50000/'
    ]  # include trailing slash
    suffix = 's2w'

plot_folder += suffix + '/'
save_folders = [root_folder + save_folder for save_folder in save_folders]

# %% mean and std plots


def load_yhat(Ro, A_star, d_min, d_max):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)

    yhat_case_all_folders = []
    for save_folder in save_folders:
        _, folders, _ = next(walk(save_folder), (None, [], None))
        for folder in folders:
            if target_string in folder:
                yhat_case = np.loadtxt(save_folder + folder + '/yhat_test.txt')
                y_case = np.loadtxt(save_folder + folder + '/y_test.txt')
                # choose only yhat within the desired distance
                yhat_case_all_folders.append(yhat_case[np.logical_and(y_case >= d_min, y_case <= d_max)])
    return np.concatenate(yhat_case_all_folders)


def yhat_mean_std(yhat_case):
    return np.mean(yhat_case), np.std(yhat_case)


# %% plot each one separately
# # define the distances to plot
# if w2w:
#     d_all = [list(range(1, 40 + 1, 3))] * len(Ro_all)
# else:
#     d_all = [list(range(10, 46 + 1, 3)), list(range(4, 40 + 1, 3)), list(range(1, 37 + 1, 3))]

# # use wingroot-to-wall instead of wingtip-to-wall in the plot
# sensor_to_wall_distance = False

# if sensor_to_wall_distance:
#     # calculate the wingroot-to-wall distance
#     wing_len_Ro = [12.367, 17.059, 20.827]
#     d_all_label = []
#     for Ro_i in range(len(d_all)):
#         d_all_label.append([round(wing_len_Ro[Ro_i] + d) for d in d_all[Ro_i]])
#     # don't share the x-axis when comparing different Ro b/c the wingroot-to-wall distances will be different
#     sharex = False
# else:
#     d_all_label = d_all
#     sharex = True

# # create folder to save plots to
# Path(plot_folder).mkdir(parents=True, exist_ok=True)

# # plot
# plt.rcParams.update({"savefig.facecolor": (1, 1, 1, 1)})  # disable transparent background
# plt.rc('font', family='serif', size=10)

# for Ro_i, Ro in enumerate(Ro_all):
#     for A_star_i, A_star in enumerate(A_star_all):
#         fig = plt.figure(figsize=(2, 2))
#         yhat_mean = np.full(len(d_all[Ro_i]), np.nan)
#         yhat_std = yhat_mean.copy()

#         for d_i, d in enumerate(d_all[Ro_i]):  # one data point for each distance
#             yhat_mean[d_i], yhat_std[d_i] = yhat_mean_std(load_yhat(Ro, A_star, d, d))

#         plt.plot(d_all_label[Ro_i], yhat_mean, color='blue', marker='o', markersize=3, linestyle=':', linewidth=1)
#         plt.fill_between(d_all_label[Ro_i], yhat_mean + yhat_std, yhat_mean - yhat_std, color='red', alpha=0.2)

#         plt.axis('square')
#         plt.xlim(0, 50)
#         plt.ylim(0, 50)
#         # plt.xlabel('True Distance (cm)')
#         # plt.ylabel('Predicted Distance (cm)')
#         plt.tight_layout()

# %% plot all in same figure
# define the distances to plot
if w2w:
    d_all = [list(range(1, 40 + 1, 3))] * len(Ro_all)
else:
    d_all = [list(range(10, 46 + 1, 3)), list(range(4, 40 + 1, 3)), list(range(1, 37 + 1, 3))]

# use wingroot-to-wall instead of wingtip-to-wall in the plot
sensor_to_wall_distance = False

if sensor_to_wall_distance:
    # calculate the wingroot-to-wall distance
    wing_len_Ro = [12.367, 17.059, 20.827]
    d_all_label = []
    for Ro_i in range(len(d_all)):
        d_all_label.append([round(wing_len_Ro[Ro_i] + d) for d in d_all[Ro_i]])
    # don't share the x-axis when comparing different Ro b/c the wingroot-to-wall distances will be different
    sharex = False
else:
    d_all_label = d_all
    sharex = True

# create folder to save plots to
Path(plot_folder).mkdir(parents=True, exist_ok=True)

# plot
plt.rcParams.update({
    "savefig.facecolor": (1, 1, 1, 1),  # disable transparent background
    "axes.titlesize": 10,
})
plt.rc('font', family='serif', size=10)

fig, axs = plt.subplots(nrows=len(A_star_all), ncols=len(Ro_all), figsize=(6, 6), sharex=True, sharey=True)
axs_right = [[[]] * len(Ro_all)] * len(A_star_all)
for Ro_i, Ro in enumerate(Ro_all):
    for A_star_i, A_star in enumerate(A_star_all):
        yhat_mean = np.full(len(d_all[Ro_i]), np.nan)
        yhat_std = yhat_mean.copy()

        for d_i, d in enumerate(d_all[Ro_i]):  # one data point for each distance
            yhat_mean[d_i], yhat_std[d_i] = yhat_mean_std(load_yhat(Ro, A_star, d, d))

        axs[A_star_i, Ro_i].plot(d_all_label[Ro_i], yhat_mean, color='blue', marker='o', markersize=3, linestyle=':', linewidth=1)
        axs[A_star_i, Ro_i].fill_between(d_all_label[Ro_i], yhat_mean + yhat_std, yhat_mean - yhat_std, color='red', alpha=0.2)

        axs[A_star_i, Ro_i].plot([0, 50], [0, 50], color='black', linewidth=0.5)  # 45 deg line

        axs[A_star_i, Ro_i].axis('square')
        axs[A_star_i, Ro_i].set_xlim(0, 45)
        axs[A_star_i, Ro_i].set_ylim(0, 45)
        # axs[A_star_i, Ro_i].set_title('Ro={}, A*={}'.format(Ro, A_star))
        if Ro == 5:
            axs_right[A_star_i][Ro_i] = axs[A_star_i, Ro_i].secondary_yaxis('right')
            axs_right[A_star_i][Ro_i].set_yticks([])
            axs_right[A_star_i][Ro_i].set_ylabel('A*={}'.format(A_star))  # , rotation=270, va='bottom')
            # axs[A_star_i][Ro_i].yaxis.set_label_position("right")
            # axs[A_star_i, Ro_i].set_ylabel('A*={}'.format(A_star))
        if A_star == 2:
            axs[A_star_i, Ro_i].set_title('Ro={}'.format(Ro))

# ticks
axs[0, 0].set_xticks([0, 10, 20, 30, 40])
axs[0, 0].set_yticks([0, 10, 20, 30, 40])

fig.supxlabel('True Distance (cm)', fontsize=10)
fig.supylabel('Predicted Distance (cm)', fontsize=10)
# axs[2, 1].set_xlabel('True Distance (cm)')
# axs[1, 0].set_ylabel('Predicted Distance (cm)')
fig.tight_layout()
fig.savefig(plot_folder + 'predictions.pdf')

# %%
plt.show()
plt.close()

# %%
