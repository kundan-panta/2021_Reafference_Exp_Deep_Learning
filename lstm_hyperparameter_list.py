from itertools import product
from Lstm_tf2_wing_wall_regression import experiment

root_folder_ = ['']

Ro_ = [5]
A_star_ = [2]

sets_val_ = [[]]
sets_test_ = [[5]]
average_window_ = [10]

lstm_layers_ = [2]
dense_hidden_layers_ = [1]
N_units_ = [16]
lr_ = [0.0001]
dropout_ = [0.2]

all_vars = [root_folder_, Ro_, A_star_, sets_val_, sets_test_, average_window_, lstm_layers_, dense_hidden_layers_, N_units_, lr_, dropout_]
all_combinations = list(product(*all_vars))

# if sets_val == sets_test, then delete that combination
for c_ind, combination in enumerate(all_combinations):
    if combination[3] == combination[4]:  # sets_val == sets_test
        all_combinations.pop(c_ind)

print("Number of hyperparameter combinations:", len(all_combinations))
# print(all_combinations)

# try all sets of parameters
for parameters in all_combinations:
    experiment(parameters)
