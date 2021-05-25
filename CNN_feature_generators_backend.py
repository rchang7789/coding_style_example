import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import random
from collections import Counter


#--------------------------------------------------------------------------------------------------
def stratified_random_resampling(combined_labels, y_label, sampling_method, seed_val):
    """
    Performs simple random sampling on each of the over-represented y-labels separately
    
    Parameters
    ----------
    1. combined_labels: 
        Combined list of all x labels only
    Ex. combined_labels = [image_1, image_2, type_label_output]
    x labels take numpy array (various dimensions are accepted) in case of stacked images or series in case of vectors
    
    2. y_labels: 
        Takes 1D numpy array. This label must be a single list of vector containing all the labels from 0 to n. Cannot take one hot-encoding labels
    
    3. sampling_method:
        Specify whether resampling method uses "oversampling" or "undersampling" 
        
        Undersampling method always uses sampling method *without* replacement for each of the respective label
        Oversampling method uses sampling method *without* replacement by default unless the required oversampling units are greater than the units in the label
        If that is the case, sampling method *with* replacement is used for that label
        
    4. seed_val:
        Takes int that corresponds to random.sample() / random.choice() functions
    
    Returns
    -------
    1. returns combined_list with orders based on the input order of the combined_labels
    ***Must takes the lists out and reconvert to numpy array***
    
    2. returns list_output containins all the data that was removed as well as its original index value
    """
    
    #Applies random sampling
    random.seed(seed_val)

    
    #Merges y_label into a single list to perform undersampling altogether
    
    combined_labels = combined_labels + [y_label]
    
    #Determine the number of y_labels
    label_val = np.unique(y_label).tolist()

    #Count the number of data in each label
    label_count = list()
    for i in range(len(label_val)):
        label_count.append((y_label == i).sum()) #numpy way of performing .count() function in list format
        
    #Determine which label has the least count
    #******************************
    if sampling_method == 'undersampling':
        min_max_label = label_count.index(min(label_count))
    elif sampling_method == 'oversampling':
        min_max_label = label_count.index(max(label_count))
        
    
    #Reorganize the list without the min label count
    label_val.remove(min_max_label)
    #label_val[min_label] = None
    
    #Create lists of lists containing label's original index value and its respective labels
    """
    Ex. Suppose we have a y_label = [0,0,1,2,2] that contains 3 different labels
    y_label would then be converted into [[0,0], [1,0], [2,1], [3,2], [4,2]] 
    where the first index within the list is the original index value and the second index
    is the y label. This is done to track random.sample() function on which label is randomly selected
    """
    y_label_index = list()
    for i in range(len(y_label)):
        y_label_index.append([i, y_label[i]])
    
    #Now separating each of the label into its own lists
    list_output = list() #This specific lists output all the labels that need to be removed with its index value
    for i in range(len(label_val)):
        current_label_list = list()
        current_label = label_val[i]
        for j in range(len(y_label_index)):
            if y_label_index[j][1] == current_label:
                current_label_list.append(y_label_index[j])
                

        #Specifies how many of the said label needs to be removed based off the min/max label count
        if sampling_method == 'undersampling':
            target_label_count = label_count[current_label] - label_count[min_max_label]
            
            #Random sampling within a label without replacement
            randomized_list = random.sample(current_label_list, target_label_count) 
  
        elif sampling_method == 'oversampling':
            target_label_count = label_count[min_max_label] - label_count[current_label]
            
            #Random sampling within a label WITH replacement if with replacement option cannot be done
            try: 
                randomized_list = random.sample(current_label_list, target_label_count) 
            except ValueError:
                print('Selected sample is larger than the population, sampling WITH replacement is used for label: ' + str(current_label_list[0][1]))
                randomized_list = random.choices(current_label_list, k=target_label_count)
                       
        list_output.append(randomized_list)


    #---Take the combined_labels and remove each of them based on its index values---
    #Combine entire lists into a single list. If it is a binary label, then processed_list = list_output
    processed_list = list()
    for i in range(len(label_val)):
        processed_list.extend(list_output[i])
    
    #The lists must be sorted in reverse order so that when xlabels are removed, it is not affecting its index value
    processed_list.sort(reverse = True)
    
    #Deleting all the available xlabels and ylabels
    final_output = list()
    for i in range(len(combined_labels)):
        target_label = combined_labels[i]
        target_label = target_label.tolist()
        
        if sampling_method == 'undersampling':
            for j in tqdm(range(len(processed_list))):
                del target_label[processed_list[j][0]]
            final_output.append(target_label)
            
        elif sampling_method == 'oversampling':
            for j in tqdm(range(len(processed_list))):
                #Insert(index position, insert value)
                target_label.insert(processed_list[j][0], target_label[processed_list[j][0]])
            final_output.append(target_label)

    #Ouput Summary
    print('\n\n* Resampling complete * | Method used: ' + str(sampling_method))
    print('Original dataset count: ' + str(Counter(y_label)))
    
    #final_output's last index is always the y_label
    y_train_rs = np.array(final_output[len(final_output)-1])
    print('Resampled dataset count: ' + str(Counter(y_train_rs)))
    
    return final_output, list_output
    


    
data_slice_range = 20  #aka from C20
def x_label_image_horizontal_dim(df_raw_pd, corr_type, *pearson_output):
    image_horizontal_dim = 10  #10 # w 30 min time frame
    slice_increment_increase = 100  #~15
    global data_slice_range  #Global setting
    original_data_slice_range = data_slice_range #used to later change the value back to original value
    
    y_fixed = df_raw_pd
    if corr_type == 'pearson':
        for i in range(image_horizontal_dim):
            slope_output, corr_output = trend().weighted_pearson_corr(y_fixed, custom_weight_val().
                                        linearly_weighted_slice(0.75, 1, True)) #True), True)<<Locally sliced True here
            if pearson_output == ('corr',):
                df_raw_pd = pd.concat([df_raw_pd, pd.DataFrame(corr_output)], axis = 1)
            elif  pearson_output == ('slope',):
                df_raw_pd = pd.concat([df_raw_pd, pd.DataFrame(slope_output)], axis = 1)
            else:
                print('\n *** Unrecognized input ***')
                break
            data_slice_range += slice_increment_increase
        
        df_raw_pd = df_raw_pd.dropna()
        df_raw_image_np = np.array(df_raw_pd)
        
    else:            
        for i in range(image_horizontal_dim):
            if corr_type == 'spearman':
                corr_output = trend().spearman_rank_corr(y_fixed)
            elif corr_type == 'kendall':
                corr_output = trend().kendall_tau_corr(y_fixed)
            elif corr_type == 'entropy':
                corr_output = trend().relative_entropy(y_fixed, True)
            elif corr_type == 'simple moving average':
                corr_output = trend().simple_moving_average(y_fixed)
            elif corr_type == 'simple moving average cross difference':
                corr_output = trend().simple_moving_average_cross_diff(y_fixed, 0.5)              
            else:
                print('\n *** Unrecognized input ***')
                break
                
            df_raw_pd = pd.concat([df_raw_pd, pd.DataFrame(corr_output)], axis = 1)
            data_slice_range += slice_increment_increase

        df_raw_pd = df_raw_pd.dropna()
        df_raw_image_np = np.array(df_raw_pd)
    
    data_slice_range = original_data_slice_range
    return df_raw_image_np
    

#df_raw_image_np = x_label_image_horizontal_dim(df_raw_pd, 'pearson', 'slope')


def positive_negative_dual_channel_separator(df_raw_image_np):
    #Splits existing raw image into two images, one is positive and the other is negative
    #Creates dual color channels in 2D imag
    df_raw_image_np_positive_ch = df_raw_image_np.copy()
    df_raw_image_np_negative_ch = df_raw_image_np.copy()

    #Removing all negative val 
    df_raw_image_np_positive_ch[:,1:][df_raw_image_np_positive_ch[:,1:] <= 0]  = 0
    
    #Removing all positive val + negates the negative values
    df_raw_image_np_negative_ch[:,1:][df_raw_image_np_negative_ch[:,1:] >= 0]  = 0
    #df_raw_image_np_negative_ch[:,1:][df_raw_image_np_negative_ch[:,1:] < 0]  *= -1  #Negates the negative value
    
    return df_raw_image_np_positive_ch, df_raw_image_np_negative_ch
    
    
#df_raw_image_np_positive_ch, df_raw_image_np_negative_ch = positive_negative_dual_channel_separator(df_raw_image_np)


#**************************** MIN MAX SCALE ****************************
def min_max_scale_2DCNN(df_raw_image_np, feature_range_start, feature_range_end):
    scaler = MinMaxScaler(feature_range=(feature_range_start, feature_range_end))
    scaler.fit_transform(df_raw_image_np[:int(len(df_raw_image_np)*0.8)][:,1:])
    
    x_label_temp = scaler.transform(df_raw_image_np[:,1:])
    
    y_label_temp = df_raw_image_np[:,0].reshape(-1,1)
    df_raw_image_np = np.concatenate((y_label_temp, x_label_temp), axis = 1)
    
    return df_raw_image_np

#**********************************************************************


#____________________ Creating x label image of n x m dim _____________________
def x_label_image_vertical_dim(df_raw_image_np, *min_max_scale):
    #Takes np df
    x_cnn_label = list()
    y_look_forward_val = 100
    image_vertical_dim = 10
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(image_vertical_dim, len(df_raw_image_np)-y_look_forward_val):
        #first column is reserved for y label and the rest is reserved for image creation
        
        #x label creation
        image_slice = df_raw_image_np[i-image_vertical_dim:i ,1:df_raw_image_np.shape[1]]
        
        if min_max_scale == ('local scaling',): 
            image_slice_scaled = scaler.fit_transform(image_slice) 
            image_slice = image_slice_scaled
            
        x_cnn_label.append(image_slice)
    

    x_cnn_label = np.array(x_cnn_label)
    
    x_cnn_label_tf = x_cnn_label.reshape(x_cnn_label.shape[0],x_cnn_label.shape[1],x_cnn_label.shape[2],1)
    
    return x_cnn_label_tf

#______________________________________________________________________________


def y_label_fixed_MFE_MAE_percent(df_raw_image_np):
    #Takes np df
    #NOTE: y_look_forward_val and image_vertical_dim MUST match with def x_label_image_vertical_dim
    y_cnn_label = list()
    y_look_forward_val = 100
    image_vertical_dim = 10
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        #first column is reserved for y label and the rest is reserved for image creation
        start_price = df_raw_image_np[i, 0]
        
        #Obtains forward list of price for MFE/MAE calculations
        forward_i = i
        inner_forward_price_set = []
        #Obtains a
        for j in range(y_look_forward_val):
            inner_forward_price_set.append(df_raw_image_np[forward_i+j, 0])
            
        max_price = max(inner_forward_price_set)
        min_price = min(inner_forward_price_set)
        
        #MFE_MAE_percent calculation
        
        MFE_MAE_percent = (max_price - start_price) / (max_price - min_price)
        #Graph component
        #plt.plot(inner_forward_price_set, label = MFE_MAE_percent)
        #plt.legend()
        #plt.show()
        # - - - - - - - -
        
        y_cnn_label.append(MFE_MAE_percent)
    
    #converts to categorical data
    y_cnn_label = [0 if i > 0.5 else 1 for i in y_cnn_label] 
        
    y_cnn_label = np.array(y_cnn_label)

    return y_cnn_label




#__________________________ Creating y label image _____________________________

def y_label_fixed_interval_classic(df_raw_image_np):
    #Takes np df
    #NOTE: y_look_forward_val and image_vertical_dim MUST match with def x_label_image_vertical_dim
    y_cnn_label = list()
    y_look_forward_val = 50
    image_vertical_dim = 10
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        #first column is reserved for y label and the rest is reserved for image creation

        #y label creation
        if df_raw_image_np[i ,0] < df_raw_image_np[i+y_look_forward_val ,0]:
            y_cnn_label.append(0)
        else:
            y_cnn_label.append(1)
        
    y_cnn_label = np.array(y_cnn_label)

    return y_cnn_label


#-------------------------------------------------------------------------------


def y_label_variable_interval_MFE_MAE_ratio(df_raw_image_np):
    y_look_forward_val = 50
    image_vertical_dim = 10
    #Note: contains a lot of unused variables below
    rolling_steps_taken = []
    rolling_TPSL_ratio_output = []
    rolling_TPSL_ratio_output_detailed = []

   
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        start_price = df_raw_image_np[i, 0]
        rolling_earning_inner = [1,-1] #default 1 and -1 for ratio to work properly
        rolling_ratio_inner = []
        
        i_forward = i+50
        while True:
            rolling_earning_inner.append(df_raw_image_np[i_forward+1, 0] - start_price)
            
            max_inner_price = max(rolling_earning_inner)
            min_inner_price = min(rolling_earning_inner)
    
            TP_SL_ratio = abs(max_inner_price/min_inner_price)
    
            rolling_ratio_inner.append(TP_SL_ratio)
            
            i_forward += 1
            if TP_SL_ratio > 3 or TP_SL_ratio < 0.333:
                rolling_TPSL_ratio_output.append(TP_SL_ratio)
                rolling_steps_taken.append(i_forward - i)
                rolling_TPSL_ratio_output_detailed.append(rolling_ratio_inner)
                break
    
    y_cnn_label = np.array([1 if i > 2 else 0 for i in rolling_TPSL_ratio_output])
    
    return y_cnn_label
    

#-------------------------------------------------------------------------------

def y_label_fixed_interval_partial_closing_positions(df_raw_image_np):
    y_look_forward_val = 50
    image_vertical_dim = 10
    
    rolling_earning_output_detailed = []
    max_look_forward_range = 100 #bars
    
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        start_price = df_raw_image_np[i,0]
        rolling_earning_inner = []
    
        i_forward = i
        range_inner = 0 #here it acts as for loop: for j in range(max_look_forward_range):
        while True:
            diff = df_raw_image_np[i_forward+1,0] - start_price #+1 = next hour, adjustable
            rolling_earning_inner.append(diff)
    
            i_forward += 1
            range_inner += 1 
            
            if range_inner == max_look_forward_range:
                rolling_earning_output_detailed.append(rolling_earning_inner)
                break
    
    rolling_earning_output = [sum(i) for i in rolling_earning_output_detailed]
    
    #Creating binary label for the said output
    sum_price_cut_off = 0
    y_cnn_label = np.array([0 if i > sum_price_cut_off else 1 for i in rolling_earning_output])
    
    print('\n\nPartial closing positions label 0/1 label Percentage: ' + str(round(y_cnn_label.tolist().count(0)/len(y_cnn_label),4)))
    
    return y_cnn_label



#-------------------------------------------------------------------------------


def y_label_fixed_interval_MFE_MAE_ratio(df_raw_image_np):
    y_look_forward_val = 50
    image_vertical_dim = 10
    
    rolling_MFE_MAE_ratio_output = []
    max_look_forward_range = 100 #bars
    
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        start_price = df_raw_image_np[i,0]
        rolling_earning_inner = []

        i_forward = i
        range_inner = 0
        while True:
            diff = df_raw_image_np[i_forward+1,0] - start_price #+1 = next hour, adjustable
            rolling_earning_inner.append(diff)
    
            i_forward += 1
            range_inner += 1 
            
            if range_inner == max_look_forward_range:
                MFE = max(rolling_earning_inner)
                MAE = min(rolling_earning_inner)
                
                MFE_MAE_ratio = abs(MFE/MAE)
                
                #Ratio can encounter inf value when MAE is zero: such instances defaults MAE to 0.1
                if MFE_MAE_ratio == float('inf'):
                    MFE_MAE_ratio = MFE/0.1
                    
                rolling_MFE_MAE_ratio_output.append(MFE_MAE_ratio)
                break
    
    #Creating binary label for the said output rolling_MFE_MAE_ratio_output
    ratio_cut_off = 1 
    y_cnn_label = np.array([0 if i > ratio_cut_off else 1 for i in rolling_MFE_MAE_ratio_output])
    
    print('\n\nMFE/MAE ratio label output 0/1 label Percentage: ' + str(round(y_cnn_label.tolist().count(0)/len(y_cnn_label),4)))
    
    return y_cnn_label
    

#-------------------------------------------------------------------------------


#_________________ fixed interval time spent floating profit/loss ratio_____________________


def y_label_fixed_interval_time_spent_floating_PL_ratio(df_raw_image_np):
    y_look_forward_val = 50
    image_vertical_dim = 10
    
    rolling_time_spent_PL_ratio_output_detailed = []
    max_look_forward_range = 100 #bars
    
    for i in tqdm(range(image_vertical_dim-1, len(df_raw_image_np)-y_look_forward_val-1)):
        start_price = df_raw_image_np[i,0]
        rolling_earning_inner = []
    
        i_forward = i
        range_inner = 0
        while True:
            diff = df_raw_image_np[i_forward+1,0] - start_price #+1 = next hour, adjustable
            rolling_earning_inner.append(diff)
    
            i_forward += 1
            range_inner += 1 
            
            if range_inner == max_look_forward_range:
                #Converts the price difference into binary 'profit' and 'loss'
                
                #floating_profit_time_count = ['floating profit' if i > 0 else 'floating loss' for i in rolling_earning_inner]
                #0 indicates floating profit count whereas 1 indicates floating loss count
                floating_profit_time_count = [0 if i > 0 else 1 for i in rolling_earning_inner]
    
                rolling_time_spent_PL_ratio_output_detailed.append(floating_profit_time_count)
                break

    rolling_time_spent_PL_ratio_output = [sum(i)/max_look_forward_range for i in rolling_time_spent_PL_ratio_output_detailed]
    
    #0 value indicates an floating trade that spent all the time on profit and vice versa for value 1
    #Creating binary label for the said output where 0.5 spent about half the time on profit and other half on loss 
    sum_price_cut_off = 0.5
    y_cnn_label = np.array([0 if i < sum_price_cut_off else 1 for i in rolling_time_spent_PL_ratio_output])
    
    print('\n\nFloating PL ratio label 0/1 label Percentage: ' + str(round(y_cnn_label.tolist().count(0)/len(y_cnn_label),4)))
    
    return y_cnn_label
    

#-------------------------------------------------------------------------------





#-------------------------------------------------------------------------------


def train_valid_test_split(x_label, y_label_pred):
    #________________ splitting train, validation, and test  _________________
    split = 0.2
    split_cut_off = int(len(x_label)*(1-split))
    #Remains in list format
    x_train = x_label[:split_cut_off]
    y_train = y_label_pred[:split_cut_off]
    
    #numpy converted already
    x_validation_test = x_label[split_cut_off:]
    y_validation_test = y_label_pred[split_cut_off:]
    
    #_________________________________________________________________________
    valid_test_split = 0.5
    valid_test_split_cut_off = int(len(x_validation_test)*(1-valid_test_split))
    
    x_validation = np.array(x_validation_test[:valid_test_split_cut_off])
    y_validation = np.array(y_validation_test[:valid_test_split_cut_off])
    
    x_test = np.array(x_validation_test[valid_test_split_cut_off:])
    y_test = np.array(y_validation_test[valid_test_split_cut_off:])
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test
    
