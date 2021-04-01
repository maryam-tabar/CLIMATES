import keras
import warnings
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from itosfm import ITOSFM
from tcn.tcn import compiled_tcn
from keras.layers import LSTM, TimeDistributed

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

def build_TCN_model(my_nb_filters):
    model = compiled_tcn(num_feat=1, nb_filters=my_nb_filters, max_len=None, kernel_size=2, return_sequences=True,
                         dropout_rate=0.0, lr=0.001, regression=True, num_classes=1, dilations=(1, 2, 4, 8, 16, 32),
                         nb_stacks=1)

    return model
