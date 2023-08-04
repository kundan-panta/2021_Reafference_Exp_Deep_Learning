from Lstm_tf2_wing_wall_regression import experiment

root_folder = ''  # include trailing slash
data_folder = root_folder + 'data/2021.07.28/raw/'  # include trailing slash
save_folder = root_folder + 'plots/2022.05.10_noshuffle_Te=5_sh=5/'  # include trailing slash

# Ro, A*, average_window, lstm_layers, N_units, dropout
best_params = [
    # (2, 2, 15, 2, 192, 0.5),
    # (2, 3, 15, 2, 128, 0.2),
    # (2, 4, 15, 2, 192, 0.2),
    # (3.5, 2, 15, 3, 192, 0.5),
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
    sets_test = [5]
    dense_hidden_layers = 1
    lr = 0.0005
    shuffle_seed = 5

    return (Ro, A_star, sets_val, sets_test, average_window, lstm_layers, dense_hidden_layers, N_units, lr, dropout, shuffle_seed)


all_combinations = []
for parameters in best_params:
    all_combinations.append(parameter_full(parameters))

print("Number of hyperparameter combinations:", len(all_combinations))
# print(all_combinations)

# try all sets of parameters
for parameters in all_combinations:
    experiment(data_folder, save_folder, parameters)
