import math

import keras
import kerascv
import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.src.layers import MultiHeadAttention, Conv2D, Add
from tensorflow.keras.layers import Dropout, BatchNormalization

from layers.Activations import Mish
from layers.PositionalEncoding import PositionalEncoding
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformerInputHead


def create_cnn_architecture(input_tensor, length, min_resolution=64, max_channels=64):
    x = input_tensor
    current_height = input_tensor.shape[1]  # Current height of the input image
    current_width = input_tensor.shape[2]  # Current width of the input image
    current_channels = input_tensor.shape[-1] + 1  # Adjust channels

    # Calculate total downscales possible
    total_downscales = 0
    while current_height > min_resolution and current_width > min_resolution:
        total_downscales += 1
        current_height //= 2
        current_width //= 2

    # Calculate downscale frequency
    if total_downscales > 0:
        downscale_frequency = length // total_downscales
    else:
        downscale_frequency = length  # If no downscale possible, set to total length

    # Reset current height and width for actual processing
    current_height = input_tensor.shape[1]
    current_width = input_tensor.shape[2]

    for i in range(length):
        print(f"CNN layer {i}")

        # Calculate the current number of channels
        current_channels = min(current_channels * (2 ** i), max_channels)

        # Save the input for the residual connection
        residual = x

        # Convolution to adjust the number of channels
        x = Conv2D(current_channels, (3, 3), padding='same', activation='mish')(x)
        x = Conv2D(current_channels, (3, 3), padding='same', activation='mish')(x)

        # Ensure the residual has the same number of channels
        if residual.shape[-1] != x.shape[-1]:
            residual = Conv2D(x.shape[-1], (1, 1))(residual)

        x = Add()([x, residual])

        x = BatchNormalization()(x)
        x = Mish()(x)

        x = SpatialAttention(use_skip_connection=True)(x)


        # Check if downscale_frequency is non-zero before the modulo operation
        if downscale_frequency > 0 and total_downscales > 0 and i % downscale_frequency == 0:
            if current_height > min_resolution and current_width > min_resolution:
                x = Conv2D(current_channels, (1, 1), strides=(2, 2), padding='same', activation=None)(x)
                current_height //= 2  # Update height
                current_width //= 2  # Update width

    return x


def create_dense_architecture(input_tensor, units=256, depth=3, dropout=0.2):
    x = input_tensor

    # Build the model using the same number of units for each layer
    for i in range(depth):
        # Save the input to add as a residual if necessary
        residual = x

        # Apply dense layer with a constant number of units
        x = layers.Dense(units, activation='mish')(x)
        x = Dropout(dropout)(x)

        # Add residual connection after the first layer
        if i > 0:
            x = layers.Add()([x, residual])  # Residual connection

        # Apply normalization and activation after the add operation
        x = layers.BatchNormalization()(x)
        x = layers.Activation('mish')(x)  # 'mish' is applied here

    return x


def round_to_power_of_2(number, up=False):
    """
    Rounds a given number to the nearest power of 2.

    Parameters:
    number (float or int): The number to be rounded.
    up (bool): If True, rounds up to the nearest power of 2; if False, rounds down.

    Returns:
    float or int: The rounded number.

    Raises:
    ValueError: If the number is less than or equal to zero.
    """
    if number <= 0:
        raise ValueError("The number must be greater than zero.")

    # Calculate the logarithm base 2
    log2 = math.log2(number)

    if up:
        # Round up to the next power of 2
        return 2 ** math.ceil(log2)
    else:
        # Round down to the previous power of 2
        return 2 ** math.floor(log2)


# The idea is to turn the image into a sequence of vectors that can be used to turn into text
def create_encoder_architecture(input_tensor, max_sequence_length=512, char_count=128, dense_layers=3):
    x = layers.BatchNormalization()(input_tensor)

    x = layers.Conv2D(x.shape[-1], (1, 1), padding='same', activation='mish')(x)

    x = layers.Flatten()(x)

    # reshape into sequence of 512
    x = layers.Reshape((max_sequence_length, -1))(x)

    x = create_dense_architecture(x, units=256, depth=dense_layers)

    x = PositionalEncoding()(x)

    residual = x

    x = MultiHeadAttention(num_heads=8, key_dim=x.shape[-1])(x, x)

    x = layers.Add()([x, residual])

    x = layers.BatchNormalization()(x)
    x = layers.Activation('mish')(x)

    x = layers.TimeDistributed(layers.Dense(char_count, activation='mish'))(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Instantiate the SpatialTransformerInputHead
    processing_head = SpatialTransformerInputHead(downscaling=2)(inputs)  # Ensure the output is used correctly

    attention = SpatialAttention(use_skip_connection=True)(processing_head)

    x = create_cnn_architecture(attention, 4, 64, 64)


    x = create_encoder_architecture(x, max_sequence_length, dense_layers=4)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(x)

    return Model(inputs, outputs, name='qr_model')
