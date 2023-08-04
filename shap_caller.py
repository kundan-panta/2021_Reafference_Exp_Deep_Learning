from itertools import product
from shap_multiple import shap_apply

root_folder = ''
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
save_folder = root_folder + 'plots/2022.04.16_exp_sh=50000/'  # include trailing slash

# specific hyperparameters only
# Ro, A*, average_window, lstm_layers, N_units, dropout
best_params = [
    (2, 2, 15, 2, 192, 0.5),
    (2, 3, 15, 2, 128, 0.2),
    # (2, 4, 15, 2, 192, 0.2),
    (3.5, 2, 15, 3, 192, 0.5),
    # (3.5, 3, 15, 3, 192, 0.5),
    # (3.5, 4, 15, 3, 192, 0.5),
    # (5, 2, 5, 3, 192, 0.5),
    # (5, 3, 5, 3, 192, 0.5),
    (5, 4, 15, 2, 192, 0.2)
]


def parameter_full(parameters):
    # best_params doesn't contain all required params, make the full param list here

    Ro, A_star, average_window, lstm_layers, N_units, dropout = parameters

    # remaining parameters
    sets_val = []
    sets_test = []
    dense_hidden_layers = 1
    lr = 0.0005
    shuffle_seed = 50000

    return (Ro, A_star, sets_val, sets_test, average_window, lstm_layers, dense_hidden_layers, N_units, lr, dropout, shuffle_seed)


all_combinations = []
for parameters in best_params:
    all_combinations.append(parameter_full(parameters))

print("Number of hyperparameter combinations:", len(all_combinations))
# print(all_combinations)

# try all sets of parameters
for parameters in all_combinations:
    shap_apply(data_folder, save_folder, parameters)

# create hyperparameter combinatins
# Ro_ = [3.5, 5]
# A_star_ = [2, 3, 4]

# sets_val_ = [[]]
# sets_test_ = [[]]
# average_window_ = [5, 10, 15]

# lstm_layers_ = [2, 3]
# dense_hidden_layers_ = [1]
# N_units_ = [64, 128, 192]
# lr_ = [0.0005]
# dropout_ = [0.2, 0.5]
# shuffle_seed_ = [5, 50, 500, 5000, 50000]

# all_vars = [Ro_, A_star_, sets_val_, sets_test_, average_window_, lstm_layers_, dense_hidden_layers_, N_units_, lr_, dropout_, shuffle_seed_]
# all_combinations = list(product(*all_vars))

# # if sets_val == sets_test, then delete that combination
# for c_ind, combination in enumerate(all_combinations):
#     if combination[3] == combination[4] and combination[3] != []:  # sets_val == sets_test
#         all_combinations.pop(c_ind)

# print("Number of hyperparameter combinations:", len(all_combinations))
# # print(all_combinations)

# # try all sets of parameters
# for parameters in all_combinations:
#     shap_apply(data_folder, save_folder, parameters)

#
# Ro_ = [2]
# A_star_ = [2, 3]

# all_vars = [Ro_, A_star_, sets_val_, sets_test_, average_window_, lstm_layers_, dense_hidden_layers_, N_units_, lr_, dropout_, shuffle_seed_]
# all_combinations = list(product(*all_vars))

# # if sets_val == sets_test, then delete that combination
# for c_ind, combination in enumerate(all_combinations):
#     if combination[3] == combination[4] and combination[3] != []:  # sets_val == sets_test
#         all_combinations.pop(c_ind)

# print("Number of hyperparameter combinations:", len(all_combinations))
# # print(all_combinations)

# # try all sets of parameters
# for parameters in all_combinations:
#     shap_apply(data_folder, save_folder, parameters)
