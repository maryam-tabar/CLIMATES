import warnings
import numpy as np
warnings.filterwarnings("ignore")
### This code is adapted from https://github.com/z331565360/State-Frequency-Memory-stock-prediction

def load_data(filename, step=1):
    data = np.load(filename)
    #data normalization
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data = (data - min_data) / (max_data - min_data)
    #dataset split
    train_split = 3*36 - step
    val_split = 4*36 - step
    
    x_train = data[:,:train_split]
    y_train = data[:,step:(train_split+step)]
    x_val = data[:,:val_split]
    y_val = data[:,step:(val_split+step)]
    x_test = data[:,:-step]
    y_test = data[:,step:]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, data, min_data, max_data]
