import numpy as np

def Normalize(data):
    '''
    Z-score normalization
    '''
    mean_data = np.mean(data)
    std_data = np.std(data)
    norm_data = (data-mean_data)/std_data
    return norm_data

def Normalize2(data):
    '''
    Min-Max normalization
    '''
    minValue, maxValue = np.min(data), np.max(data)
    norm_data = (data-minValue) / (maxValue-minValue)
    return norm_data.astype(np.float32)