from typing import Tuple
from tensorflow import keras
from keras.layers import Conv1D, Dropout, Conv1DTranspose
    
def create_cnn_ae(input_shape: Tuple[int, ...],
                  activation: str = 'relu',
                  kernel_size: int = 7,
                  dropout: float = 0.2,
                  filters: int = 32,
                 ) -> keras.Model:
    # define model
    inputs = keras.Input(shape=input_shape)
    z = inputs
    z = Conv1D(filters=filters, kernel_size=kernel_size, padding="same", strides=2, activation=activation)(z)
    z = Dropout(rate=dropout)(z)
    z = Conv1D(filters=filters/2, kernel_size=kernel_size, padding="same", strides=2, activation=activation)(z)
    z = Conv1DTranspose(filters=filters/2, kernel_size=kernel_size, padding="same", strides=2, activation=activation)(z)
    z = Dropout(rate=dropout)(z)
    z = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding="same", strides=2, activation=activation)(z)
    z = Conv1DTranspose(filters=1, kernel_size=kernel_size, padding="same")(z)
    return keras.Model(inputs=inputs, outputs=z, name="cnn_ae_model")
