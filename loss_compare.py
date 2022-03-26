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
save_folder = root_folder + 'plots/2022.03.25_exp_best/'  # include trailing slash

# %% mean and std plots


def load_loss(Ro, A_star):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)
    _, folders, _ = next(walk(save_folder), (None, [], None))
    for folder in folders:
        if target_string in folder:
            losses_case = np.loadtxt(save_folder + folder + '/loss_test_all.txt')
            return np.mean(losses_case), np.std(losses_case)
    return np.NaN, np.NaN


for Ro_ind, Ro in enumerate(Ro_all):
    for A_star_ind, A_star in enumerate(A_star_all):
        losses_mean[Ro_ind, A_star_ind], losses_std[Ro_ind, A_star_ind] = load_loss(Ro, A_star)

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


def load_loss_all(Ro, A_star):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)
    _, folders, _ = next(walk(save_folder), (None, [], None))
    for folder in folders:
        if target_string in folder:
            losses_case = np.loadtxt(save_folder + folder + '/loss_test_all.txt')
            return losses_case


ylim = [0, 8]
plt.tight_layout()

fig = plt.figure(figsize=[16, 4])
fig.supxlabel('Ro')
fig.supylabel('Mean Absolute Error (cm)')

for A_star_ind, A_star in enumerate(A_star_all):
    losses_boxplot = []
    for Ro_ind, Ro in enumerate(Ro_all):
        losses_boxplot.append(load_loss_all(Ro, A_star))

    plt.subplot(1, 3, A_star_ind + 1)
    plt.boxplot(losses_boxplot, showfliers=False)
    plt.xticks([1, 2, 3], A_star_all)
    plt.title('A* = {}'.format(A_star))
    plt.ylim(ylim)

fig = plt.figure(figsize=[16, 4])
fig.supxlabel('A*')
fig.supylabel('Mean Absolute Error (cm)')

for Ro_ind, Ro in enumerate(Ro_all):
    losses_boxplot = []
    for A_star_ind, A_star in enumerate(A_star_all):
        losses_boxplot.append(load_loss_all(Ro, A_star))

    plt.subplot(1, 3, Ro_ind + 1)
    plt.boxplot(losses_boxplot, showfliers=False)
    plt.xticks([1, 2, 3], Ro_all)
    plt.title('Ro = {}'.format(Ro))
    # plt.ylim(ylim)

# %%
plt.show()
plt.close()

# %%
