import keras
import warnings
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from itosfm import ITOSFM
from tcn.tcn import compiled_tcn
from keras.layers import LSTM, TimeDistributed
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn import linear_model
warnings.filterwarnings("ignore")


def build_LSTM_model(learning_rate, hidden_dim):
    model = Sequential()
    model.add(LSTM(hidden_dim, batch_input_shape=(256, None, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='linear')))
    my_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=my_opt)
    return model


def build_SFM_model(layers, freq, learning_rate):
    model = Sequential()

    model.add(ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim=freq,
        return_sequences=True))

    my_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=my_opt)

    return model

def build_TCN_model(my_nb_filters, learning_rate):
    model = compiled_tcn(num_feat=1, nb_filters=my_nb_filters, max_len=None, kernel_size=2, return_sequences=True,
                         dropout_rate=0.0, lr=learning_rate, regression=True, num_classes=1, dilations=(1, 2, 4, 8, 16, 32),
                         nb_stacks=1)

    return model

def build_rf_model(max_depth, n_estimators, x_train, y_train):
    rf = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
    rf.fit(x_train, y_train)

    return rf

def build_xgboost_model(max_depth, n_estimators, x_train, y_train):
    xgb_model = xgb.XGBRegressor(max_depth = max_depth, n_estimators = n_estimators, objective = 'reg:squarederror')
    xgb_model.fit(x_train, y_train)

    return xgb_model

def build_svr_model(kernel_type, x_train, y_train):
    svr_model = SVR(kernel = kernel_type)
    svr_model.fit(x_train, y_train)

    return svr_model

def build_lr_model(x_train, y_train):
    lr_model = linear_model.LinearRegression()
    lr_model.fit(x_train, y_train)

    return lr_model