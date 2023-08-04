import numpy as np
from os import walk
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd
# sh_all = ["5_", "50_", "500_", "5000_", "50000_"]
sh_all = [
    'plots/2022.03.25_exp_best_sh=5/',
    'plots/2022.03.25_exp_best_sh=50/',
    'plots/2022.03.25_exp_best_sh=500/',
    'plots/2022.04.14_exp_best_sh=5000/',
    'plots/2022.04.16_exp_best_sh=50000/'
]
Ro_all = [2, 3.5, 5]
A_star_all = [2, 3, 4]
Distance_all = [1,4,7,10,13,16,19,22,25,28,31,34,37,40]
#time_steps = {(2,2):172, (2,3):258, (2,4):344, (3.5,2):87, (3.5,3):131, (3.5,4):174, (5,2):57, (5,3):86, (5,4):115} #2_key_dictionary
time_steps = {(2,2):114, (2,3):172, (2,4):229, (3.5,2):58, (3.5,3):87, (3.5,4):116, (5,2):115, (5,3):172, (5,4):76} #2_key_dictionary
#Te_all = [1,2,3,4,5]
root_folder = 'plots/'  # include trailing slash
array_folder = 'plots/2022.05.09_exp_best_shap/'
#print(save_folder)
Ro_d_last = {2: 16, 3.5: 15, 5: 14}  # furthest distance from wall for each wing shape
# Ro_d_last = {2: 16, 3.5: 15, 5: 14}


def load_folder(Ro, A_star):
    target_string = 'Ro={}_A={}'.format(Ro, A_star)
    #target_string_2 = 'Te={}'.format(Te)
    _, folders, _ = next(walk(save_folder), (None, [], None))
    folder_names = []
    for folder in folders:
        if target_string in folder:
            folder_names.append(folder)
            #return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
    #return np.NaN
    return folder_names

def find_group(sh):
    target_string = 'sh={}'.format(sh)
    _, folders, _ = next(walk(root_folder), (None, [], None))
    folder_names = []
    for folder in folders:
        if target_string in folder:
            folder_names.append(folder)
            #return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
    #return np.NaN
    return folder_names

######################################################
#Loss Array Plot
######################################################
##loss_4D = np.empty([3,3,14,15]) #Ro,A*,distance,losses
##loss_4D[:] = np.NaN
##for Ro_index, Ro in enumerate(Ro_all):
##    #print(Ro)
##    for A_star_index, A_star in enumerate(A_star_all):
##        #print(A_star)
##        #print(load_loss(Ro, A_star))
##        loss_array = []
##        data_folder = save_folder + load_folder(Ro, A_star)[0]
##        file_name = '\\loss_test_all.txt'
##        #print(root_folder + data_folder + file_name)
##        
##        y_test = np.loadtxt(data_folder+'/y_test.txt')
##        loss_val = np.loadtxt(data_folder+file_name) # load file contain data with 3 sample * 14 distance * 5 times format
##        #print(loss_val)
##        #print('\n')
##        loss_val_d = np.zeros_like(Distance_all, dtype=float) #create zeros array with same size as distane all
##        for d_index, d in enumerate(Distance_all):
##            loss_val_d = loss_val[y_test == d]
##            #print(loss_val_d)
##            loss_array  = np.append(loss_array,loss_val_d)
##        loss_4D[Ro_index, A_star_index,:14, :] = np.reshape(loss_array,(14,15))
##np.save(root_folder+'4D_loss_save.npy',loss_4D)



loss_5D = np.empty([5,3,3,14,15]) #Shuffle, Ro,A*,distance,losses
loss_5D[:] = np.NaN

for sh_index, sh in enumerate(sh_all):
    #print('sh = ' + str(sh))
    # save_folder = root_folder + find_group(sh)[0]+'/'
    save_folder = sh
    for Ro_index, Ro in enumerate(Ro_all):
        #print('Ro = '+ str(Ro))
        for A_star_index, A_star in enumerate(A_star_all):
            #print('A* = '+ str(A_star))
            loss_array = []
            data_folder = save_folder + load_folder(Ro, A_star)[0]
            file_name = '\\loss_test_all.txt'
            #print(root_folder + data_folder + file_name)
            
            y_test = np.loadtxt(data_folder+'/y_test.txt')
            loss_val = np.loadtxt(data_folder+file_name) # load file contain data with 3 sample * 14 distance * 5 times format
            #print(loss_val)
            #print('\n')
            
            loss_val_d = np.zeros_like(Distance_all, dtype=float) #create zeros array with same size as distane all
            for d_index, d in enumerate(Distance_all):
                loss_val_d = loss_val[y_test == d]
                #print(loss_val_d)
                loss_array  = np.append(loss_array,loss_val_d)
            loss_5D[sh_index,Ro_index, A_star_index,:14, :] = np.reshape(loss_array,(14,15))
Path(array_folder).mkdir(parents=True, exist_ok=True)
np.save(array_folder+'5D_loss_save.npy',loss_5D)
            
######################################################
#SHAP Array Plot
######################################################
##shap_5D = np.empty([3,3,210,344,7]) #Ro, A*,sample*distance,time_step,force_torque_angle
##shap_5D[:] = np.NaN
##for Ro_index, Ro in enumerate(Ro_all):
##    print(Ro)
##    for A_star_index, A_star in enumerate(A_star_all):
##        print(A_star)
##        shap_array = []
##        data_folder = save_folder + load_folder(Ro,A_star)[0]
##        file_name = '/shap.npy'
##        #print(save_folder + data_folder + file_name)
##
##        y_test = np.loadtxt(data_folder+'/y_test.txt')
##        shap_val = np.load(data_folder + file_name)
##        print(shap_val.shape)
##        #print('\n')
##        
##        shap_val_d = np.zeros_like(Distance_all, dtype=float) #create zeros array with same size as distane all
##        for d_index, d in enumerate(Distance_all):
##            shap_val_d = shap_val[y_test == d]
##            #print(loss_val_d)
##            shap_array  = np.append(shap_array,shap_val_d)
##        shap_5D[Ro_index, A_star_index,: ,:time_steps[(Ro,A_star)] ,:] = np.reshape(shap_array,(210,time_steps[(Ro,A_star)],7))
##        #print(shap_5D)
##np.save(root_folder+'5D_shap_save.npy',shap_5D)


shap_6D = np.empty([5,3,3,210,344,7]) # Shuffle, Ro, A*,sample*distance,time_step,force_torque_angle
shap_6D[:] = np.NaN
for sh_index, sh in enumerate(sh_all):
    #print('sh = ' + str(sh))
    # save_folder = root_folder + find_group(sh)[0]+'/'
    save_folder = sh
    for Ro_index, Ro in enumerate(Ro_all):
        #print('Ro = '+ str(Ro))
        for A_star_index, A_star in enumerate(A_star_all):
            #print('A* = '+ str(A_star))
            shap_array = []
            data_folder = save_folder + load_folder(Ro,A_star)[0]
            file_name = '/shap.npy'
            #print(save_folder + data_folder + file_name)

            y_test = np.loadtxt(data_folder+'/y_test.txt')
            shap_val = np.load(data_folder + file_name)
            #print(shap_val.shape)
            #print('\n')
            
            shap_val_d = np.zeros_like(Distance_all, dtype=float) #create zeros array with same size as distane all
            for d_index, d in enumerate(Distance_all):
                shap_val_d = shap_val[y_test == d]
                #print(loss_val_d)
                shap_array  = np.append(shap_array,shap_val_d)
            shap_6D[sh_index,Ro_index, A_star_index,: ,:time_steps[(Ro,A_star)] ,:] = np.reshape(shap_array,(210,time_steps[(Ro,A_star)],7))
            #print(shap_5D)
Path(array_folder).mkdir(parents=True, exist_ok=True)
np.save(array_folder+'6D_shap_save.npy',shap_6D)          
######################################################

######################################################
##Loss_array = np.load(root_folder+'4D_loss_save.npy') #Ro,A*,Te,distance,losses
##
##anova_array = []
##for Ro_ind, Ro in enumerate(Ro_all):
##    for A_star_ind, A_star in enumerate(A_star_all):
##        #anova_mean = np.nanmean(Loss_array[Ro_ind,A_star_ind,],axis = (0,1))
##        anova_mean = Loss_array[Ro_ind,A_star_ind,:,:].flat
##        for i in anova_mean:
##            anova_line= [Ro,A_star,i]
##            #print(anova_line)
##            anova_array = np.append(anova_line,anova_array)
##anova_array = np.reshape(anova_array,(1890,3))
###print(anova_mean)
##print(anova_array.shape) #(1890*3)
##np.savetxt('loss_anova.csv', anova_array, delimiter=',',header = 'Ro,A,Losses',comments = '')
###np.savetxt('loss_anova.csv', anova_array, delimiter=',') #no header version
########################################################




######################################################## 
######################################################## Trash ########################################################
######################################################## 
###################################################### Trash
##def load_loss(Ro, A_star,Te):
##    target_string = 'Ro={}_A={}'.format(Ro, A_star)
##    target_string_2 = 'Te={}'.format(Te)
##    _, folders, _ = next(walk(save_folder), (None, [], None))
##    folder_names = []
##    for folder in folders:
##        if target_string in folder and target_string_2 in folder:
##            folder_names.append(folder)
##            #return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
##    #return np.NaN
##    return folder_names
###################################################### Trash
##def load_loss(Ro, A_star):
##    target_string = 'Ro={}_A={}'.format(Ro, A_star)
##    #target_string_2 = 'Te={}'.format(Te)
##    _, folders, _ = next(walk(save_folder), (None, [], None))
##    folder_names = []
##    for folder in folders:
##        if target_string in folder:
##            folder_names.append(folder)
##            #return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
##    #return np.NaN
##    return folder_names
###print(load_loss(2,2))
##
##def load_shap(Ro, A_star):
##    target_string = 'Ro={}_A={}'.format(Ro, A_star)
##    _, folders, _ = next(walk(save_folder), (None, [], None))
##    folder_names = []
##    for folder in folders:
##        if target_string in folder:
##            folder_names.append(folder)
##            #return np.loadtxt(save_folder + folder + '/loss_val_total.txt')
##    #return np.NaN
##    return folder_names
###################################################### Trash
##array_5D = np.empty([3,3,5,16,14]) #Ro,A*,Te,distance,losses
##array_5D[:] = np.NaN
##for Ro_index, Ro in enumerate(Ro_all):
##    #print(Ro)
##    for A_star_index, A_star in enumerate(A_star_all):
##        #print(A_star)
##        for Te_index, Te in enumerate(Te_all):
##            #print(Te)
##            print(load_loss(Ro, A_star, Te))
##            data_folder = save_folder + load_loss(Ro, A_star, Te)[0]
##            file_name = '\\loss_test_all.txt'
##            print(root_folder + data_folder + file_name)
##
##            loss_array = np.loadtxt(data_folder+file_name)
##            array_5D[Ro_index, A_star_index, Te_index, :Ro_d_last[Ro], :] = np.reshape(loss_array,(Ro_d_last[Ro],14)) #16 row 14 column
##np.save(root_folder+'5D_loss_save.npy',array_5D)
###################################################### Trash



########################################################
##array_new= []
##Ro = 2
##for A_star in range(2,len(A_star_all)+2): # could be expanded into A* and Te
##               
##    root_folder = 'C:\\Users\\zzywi\\Downloads\\'  # include trailing slash
##    data_folder = '2022.02.09_test_sets\\Ro='+ str(Ro) + '_A='+ str(A_star)+'_Tr=1,2,3,4_Val=_Te=5_in=0,1,2,3,4,5_bl=None_Ne=1_Ns=1_win=10_2L1D16_lr=0.0001_dr=0.2_recdr=0.0\\'  # include trailing slash
##    file_name = '\loss_test_all.txt'
##    print(root_folder + data_folder + file_name)  
##    f = open(root_folder + data_folder + file_name, "r")
##    loss_array = [] # array for storing all losses in one flie
##    
##    for loss in f: # loop through every value in the file
##        loss_array.append(loss)#put them into a array
##    f.close()
    
##    loss_array = np.loadtxt(root_folder + data_folder + file_name)
##    
##    # loss_np = np.array(loss_array) #convert to np array
##    array_2D = np.reshape(loss_array,(16,14)) #16 row 14 column
##       
##    if not array_new: #if the array setup in the beginning is emptly
##        array_new = array_2D
##        print("Array is empty")
##    else:
##        print("List has stuff")
##        arry_old = array_new 
##        array_new =np.dstack([array_old,array_2D]) # this array should be stacking over the depth
##
###covert the array to txt file
##for row in range(0,16+1): # number of distance
##    for depth in range(0,3+1): # number of txt file stored 3~?
##        for column in range(0,14+1): # 14 loss at each distance
##            txt_file.write(array_new[depth,row,column])
##
###4D methond
##np.save(root_folder+'test_4D_save.npy',array_new)
##
###drawing the plot
##
########################################################## Trash
