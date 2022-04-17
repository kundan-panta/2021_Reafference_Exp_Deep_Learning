# %%
from os import walk
from itertools import product
import numpy as np
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file

# %% make list of all folders to get the models from
root_folder = ''  # include trailing slash
save_folder = root_folder + 'plots/2022.03.21_exp_all/'  # include trailing slash
save_folder_best = root_folder + 'plots/2022.03.25_exp_best/'

# %% pull all sub-folder names
models = []
_, subfolders, _ = next(walk(save_folder), (None, [], None))
models += subfolders

# %% make list of all hyperparameters being varied
Ro_ = [2, 3.5, 5]
A_star_ = [2, 3, 4]

average_window_ = [5, 10, 15]
lstm_layers_ = [2, 3]
N_units_ = [64, 128, 192]
dropout_ = [0.2, 0.5]

all_vars = [Ro_, A_star_, average_window_, lstm_layers_, N_units_, dropout_]
all_combinations = list(product(*all_vars))

# %% make sure there are no duplicate model names and the number of models is correct
n_combinations = len(all_combinations)
assert n_combinations == len(set(all_combinations))
assert n_combinations == len(models)

# %% get hyperparameters with min loss for each Ro and A*
losses_min_Ro_A_idx = np.zeros((len(Ro_), len(A_star_)), dtype=int)
losses_max_Ro_A_idx = np.zeros((len(Ro_), len(A_star_)), dtype=int)
losses_min_Ro_A = np.zeros((len(Ro_), len(A_star_)))
losses_max_Ro_A = np.zeros((len(Ro_), len(A_star_)))

for Ro_idx, Ro in enumerate(Ro_):
    for A_idx, A_star in enumerate(A_star_):
        # make array storing loss for all combinations
        losses = np.full(n_combinations, np.nan)

        # in a loop, convert them into file names
        for comb_idx, comb in enumerate(all_combinations):
            if comb[0] != Ro or comb[1] != A_star:
                continue

            target_string_1 = 'Ro={}_A={}'.format(comb[0], comb[1])
            target_string_2 = 'win={}'.format(comb[2])
            target_string_3 = '{}L1D{}'.format(comb[3], comb[4])
            target_string_4 = 'dr={}'.format(comb[5])

            for model in models:
                if (target_string_1 in model) and (target_string_2 in model) and (target_string_3 in model) and (target_string_4 in model):
                    # load validation losses and store the average into the array
                    losses[comb_idx] = np.mean(np.loadtxt(save_folder + model + '/loss_val_all.txt'))

            assert (comb[0] != Ro or comb[1] != A_star) or not(np.isnan(losses[comb_idx]))  # make sure model is found

        # for each Ro and A*, get the indices of the lowest loss
        loss_min_idx = np.nanargmin(losses)
        loss_max_idx = np.nanargmax(losses)
        losses_min_Ro_A_idx[Ro_idx, A_idx] = loss_min_idx
        losses_max_Ro_A_idx[Ro_idx, A_idx] = loss_max_idx
        losses_min_Ro_A[Ro_idx, A_idx] = losses[loss_min_idx]
        losses_max_Ro_A[Ro_idx, A_idx] = losses[loss_max_idx]

        # convert loss into hyperparameters
        print("Min:")
        print(all_combinations[loss_min_idx])
        print('Loss =', losses[loss_min_idx])

        print("Max:")
        print(all_combinations[loss_max_idx])
        print('Loss =', losses[loss_max_idx])

        print()

# %% copy the best models to another directory
for comb_idx in losses_min_Ro_A_idx.flatten():
    comb = all_combinations[comb_idx]

    target_string_1 = 'Ro={}_A={}'.format(comb[0], comb[1])
    target_string_2 = 'win={}'.format(comb[2])
    target_string_3 = '{}L1D{}'.format(comb[3], comb[4])
    target_string_4 = 'dr={}'.format(comb[5])

    for model in models:
        if (target_string_1 in model) and (target_string_2 in model) and (target_string_3 in model) and (target_string_4 in model):
            copy_tree(save_folder + model, save_folder_best + model)
            copy_file(save_folder + model + '.svg', save_folder_best)

# %%
