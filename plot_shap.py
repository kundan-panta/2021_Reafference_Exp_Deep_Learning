# %% RUN zhiyu_loss_compile_2.py FIRST
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]

root_folder = 'plots/2022.05.09_exp_best_shap/'  # include trailing slash
plot_folder = 'plots/2022.05.09_shap/'

# %% needed variables
shap_array = np.load(root_folder + '6D_shap_save.npy')  # Ro, A*,sample*distance, time_step, FTA
d_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
time_steps = {(2, 2): 114, (2, 3): 172, (2, 4): 229, (3.5, 2): 58, (3.5, 3): 87, (3.5, 4): 116, (5, 2): 115, (5, 3): 172, (5, 4): 76}  # 2_key_dictionary
avg_wins = {(2, 2): 15, (2, 3): 15, (2, 4): 15, (3.5, 2): 15, (3.5, 3): 15, (3.5, 4): 15, (5, 2): 5, (5, 3): 5, (5, 4): 15}  # 2_key_dictionary
t_s = 0.005

# %%
# create folder to save plots to
Path(plot_folder).mkdir(parents=True, exist_ok=True)

# plot formatting
plt.rcParams.update({
    "savefig.facecolor": (1, 1, 1, 1),  # disable transparent background
    "axes.titlesize": 10,
})
plt.rc('font', family='serif', size=10)

# %%


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
gradient = np.linspace(0, 1, len(d_all))
cmap = cm.get_cmap('viridis')
axs_right = [[[]] * len(Ro_all)] * len(A_star_all)

i = 0
for Ro_ind, Ro in enumerate(Ro_all):
    for A_star_ind, A_star in enumerate(A_star_all):
        # shap_timeseries = np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,:,:,:],axis = (0,2))) #average all distance
        # shap_timeseries = np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,0,:,:],axis = 1)) #At specific distance x = 1
        # FTA_sum_absolute= np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,:,:,:],axis = 2))
        FTA_sum_absolute = np.absolute(np.nansum(shap_array[:, Ro_ind, A_star_ind, :, :, :], axis=3))
        for d_ind, d in enumerate(d_all):
            # shap_timeseries_d = np.nanmean(FTA_sum_absolute[:, 0 + 15 * d_ind:15 + 15 * d_ind, :], axis=(0, 1))
            # plot furthest distances first
            shape1 = FTA_sum_absolute.shape[1]
            shap_timeseries_d = np.nanmean(FTA_sum_absolute[:, shape1 - 15 * (d_ind + 1): shape1 - 15 * d_ind, :], axis=(0, 1))

            axs[A_star_ind, Ro_ind].plot(np.linspace(0, 1, time_steps[(Ro, A_star)]), norm(shap_timeseries_d[0:time_steps[(Ro, A_star)]]), color=cmap(gradient[d_ind]))

            # d = 13 # the count of distance wanted 1 - 14
            # d_ind = d - 1 # the index of distance wanted

        # mark stroke reversal
        axs[A_star_ind, Ro_ind].axvline(0.25, ymin=0, ymax=1, color='black', linestyle=':', linewidth=2)
        axs[A_star_ind, Ro_ind].axvline(0.75, ymin=0, ymax=1, color='black', linestyle=':', linewidth=2)

        # axs[A_star_ind, Ro_ind].set_title('Ro=' + str(Ro) + ', ' + 'A*=' + str(A_star))
        # axs[A_star_ind, Ro_ind].yaxis.set_visible(False)
        axs[A_star_ind, Ro_ind].set_yticks([0, 1])

        # A* label on right
        if Ro == Ro_all[-1]:
            axs_right[A_star_ind][Ro_ind] = axs[A_star_ind, Ro_ind].secondary_yaxis('right')
            axs_right[A_star_ind][Ro_ind].set_yticks([])
            axs_right[A_star_ind][Ro_ind].set_ylabel('A*={}'.format(A_star))  # , rotation=270, va='bottom')
            # axs[A_star_ind][Ro_ind].yaxis.set_label_position("right")
            # axs[A_star_ind, Ro_ind].set_ylabel('A*={}'.format(A_star))
        if A_star == A_star_all[0]:
            axs[A_star_ind, Ro_ind].set_title('Ro={}'.format(Ro))

        i = i + 1

# ticks
axs[0, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])
axs[0, 0].set_xticklabels([0, 0.25, 0.5, 0.75, 1])

axs[2, 1].set_xlabel('Time (normalized)')
axs[1, 0].set_ylabel('|SHAP|')
# fig.supylabel('|SHAP|')
fig.tight_layout()

# add color bar
# vertical
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.875, 0.090575, 0.025, 0.855])  # xloc, yloc, width, height
# # cax.autoscale(tight=True)
# make_axes_locatable(cax)
# mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=gradient, orientation="vertical")
# # cax.set_yticks([0, 1])
# # cax.set_yticklabels(['{:.0f}'.format(d_all[d_i]) for d_i in [0, -1]])
# cax.set_yticks(gradient)
# cax.set_yticklabels(np.flip(np.round(d_all).astype(int)))
# cax.set_ylabel('Wingtip-to-Wall Distance')
# cax.tick_params(size=0)

# horizontal
fig.subplots_adjust(bottom=0.2)
cax = fig.add_axes([0.06, 0.075, 0.885, 0.025])  # xloc, yloc, width, height
# cax.autoscale(tight=True)
make_axes_locatable(cax)
mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=gradient, orientation="horizontal")
# cax.set_yticks([0, 1])
# cax.set_yticklabels(['{:.0f}'.format(d_all[d_i]) for d_i in [0, -1]])
cax.set_xticks(gradient)
cax.set_xticklabels(np.flip(np.round(d_all) * 3 - 2).astype(int))
cax.invert_xaxis()
cax.set_xlabel('Wingtip-to-Wall Distance')
cax.tick_params(size=0)

# %%
fig.savefig(plot_folder + 'shap_time.eps')
plt.show()
plt.close()

# %%
