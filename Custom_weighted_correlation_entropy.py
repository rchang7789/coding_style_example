import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import spearmanr, kendalltau, entropy
from sklearn.preprocessing import MinMaxScaler
import math



#---------------------------------------------------------------------------
class para_tune():
    def __init__(self):
        global data_slice_range
        try: 
            self.data_slice_range = data_slice_range
        except NameError:
            self.data_slice_range = 100

class custom_weight_val(para_tune):
    """ Unless data_slice_range is declared as a global var, the default value is 100 """
    def __init__(self):
        super().__init__()
        
    def equally_weighted_slice(self):
        equally_weighted_val = np.repeat(1, self.data_slice_range)
        return equally_weighted_val
    
    def linearly_weighted_slice(self, least_recent_weight_val, most_recent_weight_val, higher_weight_on_recent_data):
        #Recent data weighted
        """ Weighted val is linearly interpolated depending on your starting weight 
        (least recent data) and ending weight (most recent data). Ideally between 0 and 1"""
        
        linearly_weighted_step = (most_recent_weight_val-least_recent_weight_val)/self.data_slice_range
        linearly_weighted_val = np.arange(least_recent_weight_val, most_recent_weight_val, linearly_weighted_step)
        #Invert the weight if the least recent data weight val is greater than that of the
        #most recent data weight val
        if higher_weight_on_recent_data == False:
            linearly_weighted_val = linearly_weighted_val[::-1]
        
        return linearly_weighted_val
        
    def logarithmically_weighted_slice(self, scale_start_weight, scale_end_weight, lamda, higher_weight_on_recent_data):
        """ Weight is based on the cumulative distribution function of the exponential distribution and further 
        scaled using min max scaler. \n
        lamda takes float val \n
        higher_weight_on_recent_data takes boolean val and if True, weight shifts to the least recent data"""
        
        cdf_exp_list = list()
        scaler = MinMaxScaler(feature_range=(scale_start_weight, scale_end_weight))
        
        for i in range(self.data_slice_range):
            cdf_exp_list.append(1-(math.e)**(-i*lamda))
        
        cdf_exp_scaled = scaler.fit_transform(np.array(cdf_exp_list).reshape(-1,1)) 
        cdf_exp_scaled = cdf_exp_scaled.reshape(len(cdf_exp_scaled)).tolist()
        
        if higher_weight_on_recent_data == False:
            cdf_exp_scaled = cdf_exp_scaled[::-1]
        
        #================== plotting feature =====================
        plt.title('Scale output')
        plt.plot(cdf_exp_scaled)
        plt.ylabel('Weight val')
        plt.xlabel('Data point')
        #=========================================================
        
        return cdf_exp_scaled


class trend(para_tune):
    def __init__(self):
        super().__init__()

    def weighted_pearson_corr(self, df_full_data_set, weighted_val, *min_max_scale):
        """df_full_data_set takes 1D pandas data \n
           weighted_val takes list of weighted data in list format \n
           min_max_scale (optional) indicates whether df_data_sliced is to be scaled (T/F) 
        """
           
        slope_b_list = list(np.repeat(None, self.data_slice_range))
        r_corr_list = list(np.repeat(None, self.data_slice_range))
        for i in tqdm(range(self.data_slice_range,len(df_full_data_set))):
            df_data_sliced = df_full_data_set.iloc[i-self.data_slice_range:i].to_numpy()
            
            if min_max_scale == (True,): #Optinal *arg always return tuple instead of boolean 
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_data_scaled_sliced = scaler.fit_transform(df_data_sliced.reshape(-1,1)) 
                df_data_scaled_sliced = df_data_scaled_sliced.reshape(len(df_data_scaled_sliced))
                df_data_sliced = df_data_scaled_sliced
                
            #----------------------------------------------------------------------------
            #Setting up two numpy columns where y is the actual data and x is the sequence
            y_val = df_data_sliced.reshape(-1, 1)
            x_val = np.arange(1, len(y_val)+1, 1).reshape(-1, 1)
            x_y_df_data_sliced = np.concatenate((x_val,y_val), axis = 1)
            #---------------------------------------------------------------------------
            
            stats_output = DescrStatsW(x_y_df_data_sliced, weights = weighted_val) #<<<
            r_corr = stats_output.corrcoef[0][1]
            
            #standard deviation of data with given ddof (0)
            stds = stats_output.std_ddof()
            
            #Calculating slope
            # b = r * (STDy / STDx)
            slope_b = r_corr * (stds[1]/stds[0])
            
            slope_b_list.append(slope_b)
            r_corr_list.append(r_corr)
            
        return slope_b_list, r_corr_list
    
            
    def spearman_rank_corr(self, df_full_data_set):
        #Nonparametric rank correlation
        spearman_nonparametric_list = list(np.repeat(None, self.data_slice_range)) 
        
        df_full_data_set = df_full_data_set.to_numpy()
        x_range = np.arange(1, data_slice_range+1, 1)
        for i in tqdm(range(data_slice_range, len(df_full_data_set))):
            df_data_sliced = df_full_data_set[i-data_slice_range:i]
                
            spearmanr_output = spearmanr(a = df_data_sliced, b = x_range)
            spearman_nonparametric_list.append(spearmanr_output[0])
            
        return spearman_nonparametric_list

    def kendall_tau_corr(self, df_full_data_set):
        #Nonparametric rank correlation
        kendall_nonparametric_list = list(np.repeat(None, self.data_slice_range)) 
        
        df_full_data_set = df_full_data_set.to_numpy()
        x_range = np.arange(1, data_slice_range+1, 1)
        for i in tqdm(range(data_slice_range, len(df_full_data_set))):
            df_data_sliced = df_full_data_set[i-data_slice_range:i]
                
            kendall_output = kendalltau(df_data_sliced, x_range)
            kendall_nonparametric_list.append(kendall_output[0])
            
        return kendall_nonparametric_list


    def relative_entropy(self, df_full_data_set, *inverse):
        entropy_linear_list = list(np.repeat(None, self.data_slice_range))
        
        df_full_data_set = df_full_data_set.to_numpy()
        x_range = np.arange(1, data_slice_range+1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        for i in tqdm(range(self.data_slice_range, len(df_full_data_set))):
            df_data_sliced = df_full_data_set[i-data_slice_range:i]

            #Min Max scaling
            df_data_scaled_sliced = scaler.fit_transform(df_data_sliced.reshape(-1,1)) 
            df_data_scaled_sliced = df_data_scaled_sliced.reshape(len(df_data_scaled_sliced))
            df_data_sliced = df_data_scaled_sliced
            
            relative_entropy = entropy(df_data_sliced, x_range)
            entropy_linear_list.append(relative_entropy)
            
        #Minus 1 for all the values since 0 indicates low entropy which may not work well in image CNN
        if inverse == (True,):
            entropy_linear_inversed_list = list()
            for i in range(len(entropy_linear_list)):
                try:
                    inv_val = 1-entropy_linear_list[i]
                    entropy_linear_inversed_list.append(inv_val)
                except TypeError:
                    entropy_linear_inversed_list.append(None)
            entropy_linear_list = entropy_linear_inversed_list
                    
        return entropy_linear_list
    

    def simple_moving_average(self, df_full_data_set):
        simple_moving_average_list = list(np.repeat(None, self.data_slice_range)) 
        
        df_full_data_set = df_full_data_set.to_numpy()
        for i in tqdm(range(self.data_slice_range, len(df_full_data_set))):
            df_data_sliced = sum(df_full_data_set[i-self.data_slice_range:i])/self.data_slice_range
            simple_moving_average_list.append(df_data_sliced)
            
        return simple_moving_average_list


    def simple_moving_average_cross_diff(self, df_full_data_set, second_ma_multiplier):
        simple_moving_average_cross_diff_list = list(np.repeat(None, self.data_slice_range)) 
        
        df_full_data_set = df_full_data_set.to_numpy()
        for i in tqdm(range(self.data_slice_range, len(df_full_data_set))):
            df_data_sliced_ma1 = sum(df_full_data_set[i-self.data_slice_range:i])/self.data_slice_range
            df_data_sliced_ma2 = sum(df_full_data_set[i-int(self.data_slice_range*second_ma_multiplier):i])/int(self.data_slice_range*second_ma_multiplier)
            
            df_data_sliced_diff = df_data_sliced_ma2 - df_data_sliced_ma1
            
            if df_data_sliced_ma1 == 0.0 or df_data_sliced_ma2 == 0.0:
                 simple_moving_average_cross_diff_list.append(None)
            else:
                simple_moving_average_cross_diff_list.append(df_data_sliced_diff)
            
        return simple_moving_average_cross_diff_list



