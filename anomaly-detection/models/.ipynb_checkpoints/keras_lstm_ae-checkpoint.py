from typing import Tuple
from tensorflow import keras
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
    
def create_lstm_ae(input_shape: Tuple[int, ...],
                  units: int = 64,
                  activation: str = 'relu',
                  dropout: float = 0.2
                 ) -> keras.Model:
    # define model
    (n_timesteps, n_features) = input_shape
    inputs = keras.Input(shape=input_shape)
    z = inputs
    z = LSTM(units = units)(z)
    z = Dropout(rate=dropout)(z)
    z = RepeatVector(n_timesteps)(z)
    z = LSTM(units=units, return_sequences=True)(z)
    z = Dropout(rate=dropout)(z)
    z = TimeDistributed(Dense(units = n_features))(z)
    return keras.Model(inputs=inputs, outputs=z, name="lstm_ae_model")
