import numpy as np
from os import walk
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm


import csv
import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#pip install statsmodels
sh_all = ["5_", "50_", "500_", "5000_", "50000_"]
Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]
d_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#Te_all = [1,2,3,4,5]
#time_steps = {(2,2):172, (2,3):258, (2,4):344, (3.5,2):87, (3.5,3):131, (3.5,4):174, (5,2):57, (5,3):86, (5,4):115} #2_key_dictionary
time_steps = {(2,2):114, (2,3):172, (2,4):229, (3.5,2):58, (3.5,3):87, (3.5,4):116, (5,2):115, (5,3):172, (5,4):76} #2_key_dictionary

root_folder = 'plots/2022.05.09_exp_best_shap/'  # include trailing slash
#save_folder = root_folder + '2022.03.25_exp_best/'
#print(save_folder)
Ro_d_last = {2: 16, 3.5: 15, 5: 14}  # furthest distance from wall for each wing shape

FTA = {0:'Fx', 1:'Fy',2:'Fz',3:'Tx',4:'Ty',5:'Fx',6:'Tz',7:'Angle',}
WL = {2:5, 3.5:7, 5:8}



#Anova_csv = np.array(pd.read_csv(root_folder + 'loss_anova.csv'))
#print(Anova_csv.shape)
#print(Anova_csv)


##Ro_array = Anova_csv[np.where(Anova_csv[:,0] == 2)]
##print(Ro_array.shape)
##np.savetxt('Ro_loss_anova.csv', Ro_array, delimiter=',',header = 'Ro,A,Losses',comments = '')
##Ro_anova = pd.read_csv(root_folder + 'Ro_loss_anova.csv')
##df = pd.DataFrame(Ro_anova)
##    
##cat = ols('Losses~A',data = df).fit()
##table = sm.stats.anova_lm(cat, typ=2)
##print(table)
##print("\n")


################### One way ANOVA Test Holding Ro Constant #################
##print('----------When Holding Ro constnat----------')
##Ro_anova_array = []
##for Ro_ind, Ro in enumerate(Ro_all):
##    Ro_anova = Anova_csv[np.where(Anova_csv[:,0] == Ro)] # find all the row that the first column value = Ro
##    print('Ro = {} Anova Result:'.format(Ro))
##    file_name = 'Ro={}_loss_anova.csv'.format(Ro)
##    #print(Ro_anova)
##    np.savetxt(file_name, Ro_anova, delimiter=',',header = 'Ro,A,Losses',comments = '')
##    Ro_anova = pd.read_csv(root_folder + file_name)
##    df = pd.DataFrame(Ro_anova)
##    
##    cat = ols('Losses~A',data = df).fit()
##    table = sm.stats.anova_lm(cat, typ=2)
##    print(table)
##    print("\n")
################### One way ANOVA Test Holding A* Constant #################
##print('----------When Holding A* constnat----------')
##A_star_anova_array = []
##for A_star_ind, A_star in enumerate(A_star_all):
##    A_star_anova = Anova_csv[np.where(Anova_csv[:,1] == A_star)] # find all the row that the second column value = A*
##    print('A* = {} Anova Result:'.format(A_star))
##    file_name = 'A={}_loss_anova.csv'.format(A_star)
##    #print(Ro_anova)
##    np.savetxt(file_name, A_star_anova, delimiter=',',header = 'Ro,A,Losses',comments = '')
##    A_star_anova = pd.read_csv(root_folder + file_name)
##    df = pd.DataFrame(A_star_anova)
##    
##    cat = ols('Losses~Ro',data = df).fit()
##    table = sm.stats.anova_lm(cat, typ=2)
##    print(table)
##    print("\n")


################# One way ANOVA Test Holding Ro Constant Contorl distance #################
##print('----------When Holding Ro constnat----------')
##Ro_anova_array = []
##for Ro_ind, Ro in enumerate(Ro_all):
##    Ro_anova = Anova_csv[np.where(Anova_csv[:,0] == Ro)] # find all the row that the first column value = Ro
##    
##    
##    #print(Ro_anova)
##    file_name_upper = 'Ro={}_loss_anova_upper.csv'.format(Ro)
##    file_name_lower = 'Ro={}_loss_anova_lower.csv'.format(Ro)
##
##    #distance 1-19cm
##    print('Ro = {} 1-19 cm Anova Result:'.format(Ro))
##    Ro_anova_upper = np.array_split(Ro_anova, 2)[0]  
##    np.savetxt(file_name_upper, Ro_anova_upper, delimiter=',',header = 'Ro,A,Losses',comments = '')    
##    Ro_anova_upper = pd.read_csv(root_folder + file_name_upper)
##    df = pd.DataFrame(Ro_anova_upper)
##    cat_upper = ols('Losses~A',data = df).fit()
##    table = sm.stats.anova_lm(cat_upper, typ=2)
##    print(table)
##        
##    #distance 22-40cm
##    print('Ro = {} 22-40 cm Anova Result:'.format(Ro))
##    Ro_anova_lower = np.array_split(Ro_anova, 2)[1]                    
##    np.savetxt(file_name_lower, Ro_anova_lower, delimiter=',',header = 'Ro,A,Losses',comments = '')
##    Ro_anova_lower = pd.read_csv(root_folder + file_name_lower)
##    df = pd.DataFrame(Ro_anova_lower)
##    cat_lower = ols('Losses~A',data = df).fit()
##    table = sm.stats.anova_lm(cat_lower, typ=2)
##    print(table)
##    print("\n")

############### One way ANOVA Test Holding A* Constant Contorl distance #################
##print('----------When Holding A* constnat----------')
##Ro_anova_array = []
##for A_star_ind, A_star in enumerate(A_star_all):
##    A_star_anova = Anova_csv[np.where(Anova_csv[:,1] == A_star)] # find all the row that the second column value = A
##    
##    
##    #print(A_star_anova)
##    file_name_upper = 'A_star={}_loss_anova_upper.csv'.format(A_star)
##    file_name_lower = 'A_star={}_loss_anova_lower.csv'.format(A_star)
##
##    #distance 1-19cm
##    print('A_star = {} 1-19 cm Anova Result:'.format(A_star))
##    A_star_anova_upper = np.array_split(A_star_anova, 2)[0]  
##    np.savetxt(file_name_upper, A_star_anova_upper, delimiter=',',header = 'Ro,A,Losses',comments = '')    
##    A_star_anova_upper = pd.read_csv(root_folder + file_name_upper)
##    df = pd.DataFrame(A_star_anova_upper)
##    cat_upper = ols('Losses~Ro',data = df).fit()
##    table = sm.stats.anova_lm(cat_upper, typ=2)
##    print(table)
##        
##    #distance 22-40cm
##    print('A_star = {} 22-40 cm Anova Result:'.format(A_star))
##    A_star_anova_lower = np.array_split(A_star_anova, 2)[1]                    
##    np.savetxt(file_name_lower, A_star_anova_lower, delimiter=',',header = 'Ro,A,Losses',comments = '')
##    A_star_anova_lower = pd.read_csv(root_folder + file_name_lower)
##    df = pd.DataFrame(A_star_anova_lower)
##    cat_lower = ols('Losses~Ro',data = df).fit()
##    table = sm.stats.anova_lm(cat_lower, typ=2)
##    print(table)
##    print("\n")
##

################# One way ANOVA Test Holding Ro Constant Based on Wing size #################
##print('----------When Holding Ro constnat----------')
##Ro_anova_array = []
##for Ro_ind, Ro in enumerate(Ro_all):
##    Ro_anova = Anova_csv[np.where(Anova_csv[:,0] == Ro)] # find all the row that the first column value = Ro
##    
##    file_name = 'Ro={}_loss_anova_wingsize.csv'.format(Ro)
##    
##    #Ro=2: 1-13cm(5), Ro=3.5: 1-19cm(7), Ro=5 1-22cm(8) 3*15 losses at each distance
##    print('Ro = {} winglength Anova Result:'.format(Ro))
##    Ro_anova_WL = Ro_anova[0:WL[Ro]*45]
##    #Ro_anova_WL = Ro_anova[0:7*45]  ##this is a sample of taken half 
##    np.savetxt(file_name, Ro_anova_WL, delimiter=',',header = 'Ro,A,Losses',comments = '')    
##    Ro_anova_WL = pd.read_csv(root_folder + file_name)
##    df = pd.DataFrame(Ro_anova_WL)
##    cat = ols('Losses~A',data = df).fit()
##    table = sm.stats.anova_lm(cat, typ=2)
##    print(table)
##    print("\n")






#################Two way ANOVA Test#################
##array = pd.read_csv(root_folder + 'loss_anova.csv')
##df = pd.DataFrame(array)
##
##cat = ols('Losses~Ro+A+CRo:A',data = df).fit()
##table = sm.stats.anova_lm(cat, typ=2)
##print(table)
















################ Coefficient Test ###################
##y = anova_array[:,2] #loss column
##X1 = anova_array[:,0]#Ro Column
##X2 = anova_array[:,1]#A* Column
##model = sm.OLS(y,[X1,X2])
##model_1 = sm.OLS(y,X1)
##model_2 = sm.OLS(y,X2)
##results_1 = model_1.fit()
##results_2 = model_2.fit()
##print(results_1.summary())
##print(results_2.summary())

##y = df.Losses
##X = df[['Ro','A']]
##model = sm.OLS(y,X)
##anova_table = sm.stats.anova_lm(model)
##print(anova_table)

##df = pd.get_dummies(df,columns=['Ro'],prefix='Ro',drop_first=True)
##df = pd.get_dummies(df,columns=['A'],prefix='A',drop_first=True)


######################################################################################################
################################## Shap Figure ##################################
######################################################################################################
shap_array = np.load(root_folder+'6D_shap_save.npy') #Ro, A*,sample*distance, time_step, FTA
###print(shap_array.shape)


##for Ro_ind, Ro in enumerate(Ro_all):
##    for A_star_ind, A_star in enumerate(A_star_all):
##        fig_shap_bar = plt.figure()
##        shap_abs_sum = np.nansum(np.absolute(shap_array[Ro_ind,A_star_ind,:,:,:]),axis = (0,1))# (Ro,A*,sample*distance, time_step, FTA) -> (sample*distance, time_step, FTA)
##        plt.bar(['Fx','Fy','Fz','Tx','Ty','Tz','Angle'],shap_abs_sum)
##        
##        plt.xlabel('')
##        plt.ylabel('Shapley Absolute Sum')
##        plt.title('Ro = ' + str(Ro) +' & '+ 'A* = ' + str(A_star))
##        
##        shap_max = np.where(shap_abs_sum == np.amax(shap_abs_sum))
##        print('Ro = {}, A* = {}'.format(Ro,A_star))
##        print('Maximum shapely is : {}'.format(FTA[ int(shap_max[0]) ]) )
##        print(shap_abs_sum)
##        print('\n')
##plt.show()
##plt.close(fig_shap_bar)
     

################### Time series Shapley absolute then sum #################
##i =1
##fig_shap = plt.figure()
##for Ro_ind, Ro in enumerate(Ro_all):
##    for A_star_ind, A_star in enumerate(A_star_all):
##        
##        shap_timeseries = np.nansum(np.absolute(shap_array[Ro_ind,A_star_ind,:,:,:]),axis = (0,2)) #average all distance
##        #shap_timeseries = np.nansum(np.absolute(shap_array[Ro_ind,A_star_ind,0,:,:]),axis = 1) #At specific distance x = 1
##
##        ##print(shap_timeseries)
##        ##print('\n')
##        plt.subplot(3,3,i)
##        plt.plot(list(range(time_steps[(Ro,A_star)])),shap_timeseries[0:time_steps[(Ro,A_star)]])
##        #angle_all = shap_array[Ro_ind,A_star_ind,1,:,6] #all angle data include nan 
##        #angle = angle_all[np.logical_not(np.isnan(angle_all))] #remove nan
##        #plt.plot(list(range(time_steps[(Ro,A_star)])),angle)
##        
##        plt.xlabel('Time Steps')
##        plt.ylabel('Shapley Absolute Sum')
##        plt.title('Ro = ' + str(Ro) +' & '+ 'A* = ' + str(A_star))
##        i = i + 1
##plt.show()
##plt.close(fig_shap)

################### Time series Shapley sum then absolute #################
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 3, wspace=1, hspace=1)  # workaround to have no overlap between subplots
gradient = np.linspace(0, 1, len(d_all))
cmap = cm.get_cmap('viridis')

i =0
fig_shap = plt.figure()
for Ro_ind, Ro in enumerate(Ro_all):
    for A_star_ind, A_star in enumerate(A_star_all):
        
        #shap_timeseries = np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,:,:,:],axis = (0,2))) #average all distance
        #shap_timeseries = np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,0,:,:],axis = 1)) #At specific distance x = 1
        plt.subplot(gs[i])
        #FTA_sum_absolute= np.absolute(np.nansum(shap_array[Ro_ind,A_star_ind,:,:,:],axis = 2))
        FTA_sum_absolute= np.absolute(np.nansum(shap_array[:,Ro_ind,A_star_ind,:,:,:],axis = 3))
        for d_ind,d in enumerate(d_all):
            shap_timeseries_d = np.nanmean(FTA_sum_absolute[:,0+15*d_ind:15+15*d_ind,:],axis = (0,1))
            plt.plot(list(range(time_steps[(Ro,A_star)])),shap_timeseries_d[0:time_steps[(Ro,A_star)]],color=cmap(gradient[d_ind]))
        
            #d = 13 # the count of distance wanted 1 - 14
            #d_ind = d - 1 # the index of distance wanted
        
        
        
        #print(shap_timeseries)
        #print('\n')
        #plt.subplot(3,3,i)
        
        

        
        plt.xlabel('Time Steps')
        plt.ylabel('Shapley Absolute Sum')
        plt.title('Ro = ' + str(Ro) +' & '+ 'A* = ' + str(A_star))
        i = i + 1
plt.tight_layout()
plt.show()
plt.close(fig_shap)



################# bar plot Shapley #################

######################################################################################################
################################## loss Figure ##################################
######################################################################################################
# Loss_array = np.load(root_folder+'5D_loss_save.npy') #Ro,A*,distance,losses


# ################# 3D loss vs. distance #################

# ##for Ro_ind, Ro in enumerate(Ro_all):
# ##    ax = plt.axes(projection = '3d')
# ##    ax.set_ylim(0,41)
# ##    ax.set_zlim(-5,15)
# ##
# ####    ax.set_xlim(0,5)
# ####    ax.set_ylim(0,41)
# ####    ax.set_zlim(-5,15)
# ##    for A_star_ind, A_star in enumerate(A_star_all):
# ##        mean_losses = np.nanmean(Loss_array[Ro_ind,A_star_ind,:,:],axis = 1)
# ##        std = np.nanstd(Loss_array[Ro_ind,A_star_ind,:,:],axis = 1)
# ##        x = [A_star]*14 # A star
# ##        y = list(range(1, 41, 3))
# ##        z = mean_losses
# ##        #ax.plot(x,y, z)
# ##        ax.errorbar(x,y,z,zerr=std)
# ##        ax.set_xlabel('A*')
# ##        ax.set_ylabel('Distance')
# ##        ax.set_zlabel('Losses')
# ##     
# ##    ax.set_title('Ro = ' + str(Ro))
# ##    plt.show()
# ##



# #################ploting for all Ro & A* Combination###############
# i =0
# fig_Ro_A_star_loss = plt.figure()
# plt.tight_layout()
# for Ro_ind, Ro in enumerate(Ro_all):
#     for A_star_ind, A_star in enumerate(A_star_all):
#         plt.subplot(gs[i])
        
        
#         mean_losses = np.nanmean(Loss_array[:,Ro_ind,A_star_ind,:,:],axis = (0,2))# (Ro,A*,distance,losses axis) -> (distance,loss)
#         std = np.nanstd(Loss_array[:,Ro_ind,A_star_ind,:,:],axis = (0,2))
#         plt.errorbar(list(range(1, 41, 3)),mean_losses,yerr = std,color='blue')


#         plt.legend('')
#         plt.xlabel('Distance')
#         plt.ylabel('Test Set Loss')
#         plt.title('Ro = ' + str(Ro) +' & '+ 'A* = ' + str(A_star))
#         plt.xlim([0,41])
#         plt.ylim([-5,15])
#         i = i + 1
# plt.show()
# plt.close(fig_Ro_A_star_loss)





# #####################error bar Ro vs A* ################################## updated
# ##line_types = ['ro--', 'g*-.', 'b^:']
# ##fig_A_star_loss = plt.figure()
# ##for Ro_ind, Ro in enumerate(Ro_all):
# ##    mean_losses = np.nanmean(Loss_array[Ro_ind,:,:,:],axis = (1,2))# (Ro,A*,Te,distance,losses axis) -> (A*,distance,loss)
# ##    std = np.nanstd(Loss_array[Ro_ind,:,:,:],axis = (1,2))
# ##    plt.errorbar(A_star_all,mean_losses,yerr = std, capsize=5, label='Ro={}'.format(Ro), fmt=line_types[Ro_ind])
# ##    plt.legend()   ###???###
# ##    plt.xlabel('A*')
# ##    plt.ylabel('Test Set Loss')
# ##plt.show()
# ##plt.close(fig_A_star_loss)
# ##
# ##fig_Ro_loss = plt.figure()
# ##for A_star_ind, A_star in enumerate(A_star_all):
# ##
# ##    mean_losses = np.nanmean(Loss_array[:,A_star_ind,:,:],axis = (1,2))# (Ro,A*,distance,losses) -> (Ro,distance,loss)
# ##    std = np.nanstd(Loss_array[:,A_star_ind,:,:],axis = (1,2))
# ##    plt.errorbar(A_star_all,mean_losses,yerr = std, capsize=5, label='A*={}'.format(A_star), fmt=line_types[A_star_ind])
# ##    plt.legend()   ###???###
# ##    plt.xlabel('Ro')
# ##    plt.ylabel('Test Set Loss')
# ##plt.show()
# ##plt.close(fig_Ro_loss)#, line_types[Ro_ind], label='Ro={}'.format(Ro)
# ##





# ###############ploting for comparison among all losses###############

# ##fig_A_star_loss = plt.figure()
# ##mean_losses = np.nanmean(Loss_array,axis = (0,1,3))
# ##std = np.nanstd(Loss_array,axis = (0,1,3))
# ##plt.errorbar(list(range(14)),mean_losses,yerr = std,color='red')
# ##plt.legend('')
# ##plt.xlabel('distance')
# ##plt.ylabel('Test Set Loss')
# ##plt.title('Overall Loss vs. Distance')
# ##plt.xlim([-1,16])
# ##plt.ylim([-5,20])
# ##plt.show()
# ##plt.close(fig_A_star_loss)


# #################ploting for comparison between Rossby number###############
# ##for Ro_ind, Ro in enumerate(Ro_all):
# ##    fig_Ro_loss = plt.figure()
# ##    mean_losses = np.nanmean(Loss_array[Ro_ind,:,:,:,:],axis = (0,1,3))# (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
# ##    std = np.nanstd(Loss_array[Ro_ind,:,:,:,:],axis = (0,1,3))
# ##    plt.errorbar(list(range(16)),mean_losses,yerr = std,color='blue')
# ##    plt.legend('Ro')
# ##    plt.xlabel('distance')
# ##    plt.ylabel('Test Set Loss')
# ##    plt.title('Ro = ' + str(Ro))
# ##    plt.xlim([-1,16])
# ##    plt.ylim([-5,20])
# ##plt.show()
# ##plt.close(fig_Ro_loss)

# #################ploting for comparison between A* value###############
# ##line_types = ['ro--', 'g*-.', 'r^:']
# ##for A_star_ind, A_star in enumerate(A_star_all):
# ##    fig_A_star_loss = plt.figure()
# ##    mean_losses = np.nanmean(Loss_array[:,A_star_ind,:,:,:],axis = (0,1,3))# (Ro,A*,Te,distance,losses axis) -> (Ro,Te,distance,loss)
# ##    std = np.nanstd(Loss_array[:,A_star_ind,:,:,:],axis = (0,1,3))
# ##    plt.errorbar(list(range(16)),mean_losses,yerr = std,color='green')
# ##    plt.legend('A*')
# ##    plt.xlabel('distance')
# ##    plt.ylabel('Test Set Loss')
# ##    plt.title('A* = ' + str(A_star))
# ##    plt.xlim([-1,16])
# ##    plt.ylim([-5,20])
# ##plt.show()
# ##plt.close(fig_A_star_loss)



