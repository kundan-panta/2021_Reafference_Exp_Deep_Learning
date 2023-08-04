# %%
import numpy as np
from os import walk
import matplotlib.pyplot as plt

import csv
import pandas as pd
import random

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]
Te_all = [[1], [2], [3], [4], [5]]
Te_all = [[]]
root_folder = 'plots/'  # include trailing slash
save_folder = root_folder + '2022.02.27_transformer/'
# print(save_folder)
# Ro_d_last = {2: 16, 3.5: 15, 5: 14}  # furthest distance from wall for each wing shape
Ro_d_last = {2: 14, 3.5: 14, 5: 14}
N_examples = 15  # losses at every distance


def load_loss(Ro, A_star, Te):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)
    target_string_2 = 'Te={}'.format(','.join(str(temp) for temp in Te))
    _, folders, _ = next(walk(save_folder), (None, [], None))
    folder_names = []
    for folder in folders:
        if target_string in folder and target_string_2 in folder:
            folder_names.append(folder)
            # return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
    # return np.NaN
    return folder_names


# print(load_loss(2, 2, 1))
######################################################
array_5D = np.empty([len(Ro_all), len(A_star_all), len(Te_all), max(Ro_d_last.values()), N_examples])  # Ro,A*,Te,distance,losses
array_5D[:] = np.NaN
for Ro_index, Ro in enumerate(Ro_all):
    # print(Ro)
    for A_star_index, A_star in enumerate(A_star_all):
        # print(A_star)
        for Te_index, Te in enumerate(Te_all):
            # print(Te)
            # print(load_loss(Ro, A_star, Te))
            data_folder = save_folder + load_loss(Ro, A_star, Te)[0]
            file_name = '\\loss_test_all.txt'
            # print(root_folder + data_folder + file_name)

            loss_array = np.loadtxt(data_folder + file_name)
            array_5D[Ro_index, A_star_index, Te_index, :Ro_d_last[Ro], :] = np.reshape(loss_array, (Ro_d_last[Ro], N_examples))  # 16 row 14 column
# np.save(save_folder + '5D_loss_save.npy', array_5D)

# limit distances
# array_5D = array_5D[:, :, :, :5, :]

# %%
################################## make figures ########################################

# array_5D = np.load(save_folder + '5D_loss_save.npy')
# np.mean(array_5D, axis=(0, 1, 3))
# np.std(array_5D, axis=(0, 1, 3))


###############ploting for comparison among all losses###############
##fig_A_star_loss = plt.figure()
##mean_losses = np.nanmean(array_5D,axis = (0,1,2,4))
##std = np.nanstd(array_5D,axis = (0,1,2,4))
##plt.errorbar(list(range(16)),mean_losses,yerr = std,color='red')
# plt.legend('')
# plt.xlabel('distance')
##plt.ylabel('Test Set Loss')
##plt.title('Overall Loss vs. Distance')
# plt.xlim([-1,16])
# plt.ylim([-5,20])
# plt.show()
# plt.close(fig_A_star_loss)


#################ploting for comparison between Rossby number###############
# fig_Ro_loss = plt.figure()
# for Ro_ind, Ro in enumerate(Ro_all):
#     mean_losses = np.nanmean(array_5D[Ro_ind,:,:,:,:],axis = (0,1,3))# (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
# #    std = np.nanstd(array_5D[Ro_ind,:,:,:,:],axis = (0,1,3))
# #    plt.errorbar(list(range(16)),mean_losses,yerr = std,color='blue')
# plt.legend('Ro')
# plt.xlabel('distance')
# #    plt.ylabel('Test Set Loss')
# #    plt.title('Ro = ' + str(Ro))
# plt.xlim([-1,16])
# plt.ylim([-5,20])
# plt.show()
# plt.close(fig_Ro_loss)

#################ploting for comparison between A* value###############
##line_types = ['ro--', 'g*-.', 'r^:']
# for A_star_ind, A_star in enumerate(A_star_all):
##    fig_A_star_loss = plt.figure()
# mean_losses = np.nanmean(array_5D[:,A_star_ind,:,:,:],axis = (0,1,3))# (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
##    std = np.nanstd(array_5D[:,A_star_ind,:,:,:],axis = (0,1,3))
##    plt.errorbar(list(range(16)),mean_losses,yerr = std,color='green')
# plt.legend('A*')
# plt.xlabel('distance')
##    plt.ylabel('Test Set Loss')
##    plt.title('A* = ' + str(A_star))
# plt.xlim([-1,16])
# plt.ylim([-5,20])
# plt.show()
# plt.close(fig_A_star_loss)

###############ploting for all Ro & A* Combination###############
# for Ro_ind, Ro in enumerate(Ro_all):
#     for A_star_ind, A_star in enumerate(A_star_all):
#         fig_Ro_A_star_loss = plt.figure()
#         mean_losses = np.nanmean(array_5D[Ro_ind, A_star_ind, :, :, :], axis=(0, 2))  # (Ro,A*,Te,distance,losses axis) -> (Te,distance,loss)
#         std = np.nanstd(array_5D[Ro_ind, A_star_ind, :, :, :], axis=(0, 2))
#         plt.errorbar(list(range(16)), mean_losses, yerr=std, color='blue')
#         plt.legend('')
#         plt.xlabel('distance')
#         plt.ylabel('Test Set Loss')
#         plt.title('Ro = ' + str(Ro) + ' & ' + 'A* = ' + str(A_star))
#         plt.xlim([-1, 16])
#         plt.ylim([-5, 20])
# plt.show()
# plt.close(fig_Ro_A_star_loss)


###################error bar Ro vs A* ##################################
line_types = ['ro--', 'g*-.', 'b^:']

fig_A_star_loss = plt.figure()
for Ro_ind, Ro in enumerate(Ro_all):
    mean_losses = np.nanmean(array_5D[Ro_ind, :, :, :-2, :], axis=(1, 2, 3))  # (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
    std = np.nanstd(array_5D[Ro_ind, :, :, :-2, :], axis=(1, 2, 3))
    plt.errorbar(A_star_all, mean_losses, yerr=std, capsize=5, label='Ro={}'.format(Ro), fmt=line_types[Ro_ind])
plt.legend()  # ???###
plt.xlabel('A*')
plt.ylabel('Test Set Loss')
# plt.show()
# plt.close(fig_A_star_loss)
#
fig_Ro_loss = plt.figure()
for A_star_ind, A_star in enumerate(A_star_all):
    mean_losses = np.nanmean(array_5D[:, A_star_ind, :, :-2, :], axis=(1, 2, 3))  # (Ro,A*,Te,distance,losses) -> (Ro,Te,distance,loss)
    std = np.nanstd(array_5D[:, A_star_ind, :, :-2, :], axis=(1, 2, 3))
    plt.errorbar(Ro_all, mean_losses, yerr=std, capsize=5, label='A*={}'.format(A_star), fmt=line_types[A_star_ind])
plt.legend()  # ???###
plt.xlabel('Ro')
plt.ylabel('Test Set Loss')
# plt.show()
# plt.close(fig_Ro_loss)

plt.show()


###################error bar Ro vs A* ##################################
# line_types = ['ro--', 'g*-.', 'b^:']

fig_A_star_loss = plt.figure()

mean_losses = np.nanmean(array_5D[:, :, :, :-2, :], axis=(0, 2, 3, 4))  # (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
std = np.nanstd(array_5D[:, :, :, :-2, :], axis=(0, 2, 3, 4))
plt.errorbar(A_star_all, mean_losses, yerr=std, capsize=5, fmt='b:o')

# plt.legend()  # ???###
plt.xlabel('A*')
plt.ylabel('Test Set Loss')
# plt.show()
# plt.close(fig_A_star_loss)
#
fig_Ro_loss = plt.figure()

mean_losses = np.nanmean(array_5D[:, :, :, :-2, :], axis=(1, 2, 3, 4))  # (Ro,A*,Te,distance,losses) -> (Ro,Te,distance,loss)
std = np.nanstd(array_5D[:, :, :, :-2, :], axis=(1, 2, 3, 4))
plt.errorbar(Ro_all, mean_losses, yerr=std, capsize=5, fmt='b:o')

# plt.legend()  # ???###
plt.xlabel('Ro')
plt.ylabel('Test Set Loss')
# plt.show()
# plt.close(fig_Ro_loss)

plt.show()

#####ANOVA Test####
##header = ['R']
##f = open('root_folder','w')
# , line_types[Ro_ind], label='Ro={}'.format(Ro)

# %%
