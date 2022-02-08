from itertools import product
from Lstm_tf2_wing_wall_regression import experiment

root_folder_ = ['']

Ro_ = [2]
A_star_ = [2, 3, 4]

sets_val_ = [[4]]
sets_test_ = [[5]]
average_window_ = [10]

lstm_layers_ = [2]
dense_hidden_layers_ = [1]
N_units_ = [64]
lr_ = [0.0001]
dropout_ = [0.2]

all_vars = [root_folder_, Ro_, A_star_, sets_val_, sets_test_, average_window_, lstm_layers_, dense_hidden_layers_, N_units_, lr_, dropout_]
all_combinations = list(product(*all_vars))
print("Number of hyperparameter combinations:", len(all_combinations))
# print(all_combinations[1])

# try all sets of parameters
for parameters in all_combinations:
    experiment(parameters)

# Ro_ = [2, 3.5, 5]
# A_star_ = [2, 3, 4]

# sets_val_ = [[3], [5]]
# average_window_ = [5, 10, 20]

# lstm_layers_ = [2, 3]
# dense_hidden_layers_ = [1]
# N_units_ = [16, 64]
# lr_ = [0.0001, 0.001, 0.01]
# dropout_ = [0.2]
