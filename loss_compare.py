# %%
import numpy as np
from os import walk
import matplotlib.pyplot as plt

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]

# initialize NaN array to hold the losses for all cases
losses_mean = np.full((len(Ro_all), len(A_star_all)), np.NaN)
losses_std = np.full((len(Ro_all), len(A_star_all)), np.NaN)

# assume there's only 1 folder for each case
root_folder = ''  # include trailing slash
save_folder = root_folder + 'plots/2022.03.25_exp_best_sh=5/'  # include trailing slash

# %% mean and std plots


def load_loss(Ro, A_star, d_min, d_max):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)
    _, folders, _ = next(walk(save_folder), (None, [], None))
    for folder in folders:
        if target_string in folder:
            losses_case = np.loadtxt(save_folder + folder + '/loss_test_all.txt')
            y_case = np.loadtxt(save_folder + folder + '/y_test.txt')
            # choose only losses within the desired distance
            losses_case = losses_case[np.logical_and(y_case >= d_min, y_case <= d_max)]
            return losses_case


def loss_mean_std(losses_case):
    return np.mean(losses_case), np.std(losses_case)


# distances between which to get losses
d_min = 0
d_max = 99

for Ro_ind, Ro in enumerate(Ro_all):
    for A_star_ind, A_star in enumerate(A_star_all):
        losses_mean[Ro_ind, A_star_ind], losses_std[Ro_ind, A_star_ind] = loss_mean_std(load_loss(Ro, A_star, d_min, d_max))

# make figures
line_types = ['ro--', 'g*-.', 'b^:']

fig_Ro_loss = plt.figure()
for A_star_ind, A_star in enumerate(A_star_all):
    plt.errorbar(Ro_all, losses_mean[:, A_star_ind], yerr=losses_std[:, A_star_ind], label='A*={}'.format(A_star), fmt=line_types[A_star_ind], capsize=5)
plt.legend()
plt.xlabel('Ro')
plt.ylabel('Test Set Loss')

fig_A_star_loss = plt.figure()
for Ro_ind, Ro in enumerate(Ro_all):
    plt.errorbar(A_star_all, losses_mean[Ro_ind, :], yerr=losses_std[Ro_ind, :], label='Ro={}'.format(Ro), fmt=line_types[Ro_ind], capsize=5)
plt.legend()
plt.xlabel('A*')
plt.ylabel('Test Set Loss')

# %% boxplot
ylim = [0, 8]
plt.tight_layout()

fig = plt.figure(figsize=[16, 4])
fig.supxlabel('Ro')
fig.supylabel('Mean Absolute Error (cm)')

for A_star_ind, A_star in enumerate(A_star_all):
    losses_boxplot = []
    for Ro_ind, Ro in enumerate(Ro_all):
        losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

    plt.subplot(1, 3, A_star_ind + 1)
    plt.boxplot(losses_boxplot, showfliers=False)
    plt.xticks([1, 2, 3], Ro_all)
    plt.title('A* = {}'.format(A_star))
    # plt.ylim(ylim)

fig = plt.figure(figsize=[16, 4])
fig.supxlabel('A*')
fig.supylabel('Mean Absolute Error (cm)')

for Ro_ind, Ro in enumerate(Ro_all):
    losses_boxplot = []
    for A_star_ind, A_star in enumerate(A_star_all):
        losses_boxplot.append(load_loss(Ro, A_star, d_min, d_max))

    plt.subplot(1, 3, Ro_ind + 1)
    plt.boxplot(losses_boxplot, showfliers=False)
    plt.xticks([1, 2, 3], A_star_all)
    plt.title('Ro = {}'.format(Ro))
    # plt.ylim(ylim)

# %% make loss plots with multiple rows for each distance
d_rows = [[1, 1], [4, 4], [7, 7], [10, 10], [13, 13], [16, 16], [19, 19], [22, 22], [25, 25], [28, 28], [31, 31], [34, 34], [37, 37], [40, 40]]


# %%
plt.show()
plt.close()

# %%
