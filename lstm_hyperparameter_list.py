from itertools import product
from Lstm_tf2_wing_wall_regression import experiment

root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
save_folder = root_folder + 'plots/2022.03.22_exp_zhiyu/'  # include trailing slash

Ro_ = [5]
A_star_ = [3, 4]

sets_val_ = [[]]
sets_test_ = [[]]
average_window_ = [5]

lstm_layers_ = [2]
dense_hidden_layers_ = [1]
N_units_ = [192]
lr_ = [0.0005]
dropout_ = [0.5]
shuffle_seed_ = [50]

all_vars = [Ro_, A_star_, sets_val_, sets_test_, average_window_, lstm_layers_, dense_hidden_layers_, N_units_, lr_, dropout_, shuffle_seed_]
all_combinations = list(product(*all_vars))

# if sets_val == sets_test, then delete that combination
for c_ind, combination in enumerate(all_combinations):
    if combination[3] == combination[4] and combination[3] != []:  # sets_val == sets_test
        all_combinations.pop(c_ind)

print("Number of hyperparameter combinations:", len(all_combinations))
# print(all_combinations)

# try all sets of parameters
for parameters in all_combinations:
    experiment(data_folder, save_folder, parameters)
