import keras
import time
import warnings
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed

warnings.filterwarnings("ignore")

def load_data(filename, step=1):
    day = step
    data = np.load(filename)

    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data = (data - min_data) / (max_data - min_data)
    train_split = 3*36 - day
    val_split = 4*36 - day
    
    x_train = data[:,:train_split]
    y_train = data[:,day:(train_split+day)]
    x_val = data[:,:val_split]
    y_val = data[:,day:(val_split+day)]
    x_test = data[:,:-day]
    y_test = data[:,day:]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, data, min_data, max_data]


def build_model(learning_rate,hidden_dim):
    
    model = Sequential()

    model.add(LSTM(hidden_dim, batch_input_shape=(256,None, 1), return_sequences=True))
    
    model.add(TimeDistributed(Dense(1,activation='linear')))

    start = time.time()
    
    rms = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=rms)

    return model
