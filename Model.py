import numpy as np
from keras import layers, Model
from keras.src.layers import MultiHeadAttention, Conv2D, Add, Conv3D, Conv1D
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense

from layers.Activations import Mish
from layers.PositionalEncoding import PositionalEncoding
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformer
from layers.SqueezeExcitation import SqueezeExcitation


def create_cnn_architecture(input_tensor, length, min_resolution=64, max_channels=64, attention_frequency=2,
                            use_residual=True):
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
        if i % attention_frequency == 0:
            x = SqueezeExcitation()(x)
            x = SpatialAttention(use_skip_connection=True)(x)

        # Calculate the current number of channels
        current_channels = min(current_channels * (2 ** i), max_channels)

        residual = x

        x = Conv2D(current_channels, (3, 3), padding='same', activation='mish', kernel_initializer="he_normal")(x)

        if use_residual:
            # Ensure the residual has the same number of channels
            if residual.shape[-1] != x.shape[-1]:
                residual = Conv2D(x.shape[-1], (1, 1))(residual)

            x = Add()([x, residual])

            x = Mish()(x)
            x = BatchNormalization()(x)

        # Check if downscale_frequency is non-zero before the modulo operation
        if downscale_frequency > 0 and total_downscales > 0 and i % downscale_frequency == 0:
            if current_height > min_resolution and current_width > min_resolution:
                x = Conv2D(current_channels, (1, 1), strides=(2, 2), padding='same', activation=None)(x)
                current_height //= 2  # Update height
                current_width //= 2  # Update width

    return x


def create_dense_architecture(input_tensor, units=512, depth=3, dropout=0.1):
    x = input_tensor

    # Build the model using the same number of units for each layer
    for i in range(depth):
        # Save the input to add as a residual if necessary
        residual = x

        # Apply dense layer with a constant number of units
        x = layers.Dense(units, activation='mish', kernel_initializer="he_normal")(x)
        x = Dropout(dropout)(x)

        # Add residual connection after the first layer
        if i > 0:
            x = layers.Add()([x, residual])  # Residual connection

        # Apply normalization and activation after the add operation
        x = layers.Activation('mish')(x)  # 'mish' is applied here

        x = layers.BatchNormalization()(x)

    return x


def create_attention_module(input_tensor, heads=8, depth=1, dropout=0.1):
    x = input_tensor

    for i in range(depth):
        # Save the input to add as a residual if necessary
        residual_1 = x

        # Apply multi-head attention
        x = MultiHeadAttention(num_heads=heads, key_dim=x.shape[-1])(x, x)
        x = Dropout(dropout)(x)

        x = layers.Add()([x, residual_1])
        x = layers.BatchNormalization()(x)

        residual_2 = x

        x = layers.Dense(x.shape[-1], activation='mish', kernel_initializer="he_normal")(x)
        x = Dropout(dropout)(x)
        x = layers.Dense(x.shape[-1])(x)

        x = layers.Add()([x, residual_2])
        x = layers.Activation('mish')(x)

        x = layers.BatchNormalization()(x)

    return x


# The idea is to turn individual pixels into a sequence of embeddings
def cnn_to_sequence(input_tensor, max_sequence_length=512, feature_length=128):
    x = layers.BatchNormalization()(input_tensor)

    # eventually we might want to change this to patch extraction

    x = layers.Conv2D(x.shape[-1], (1, 1), padding='same', activation='mish', kernel_initializer="he_normal")(x)

    x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

    x = layers.Conv1D(filters=feature_length, kernel_size=1, padding='valid', activation='mish',
                      kernel_initializer="he_normal")(x)

    # Get initial input length

    x = layers.TimeDistributed(layers.Dense(feature_length * 2, activation="mish", kernel_initializer="he_normal"))(x)

    x = PositionalEncoding()(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    spatial_transformer = SpatialTransformer()(inputs)

    x = create_cnn_architecture(spatial_transformer, 4, 128, 64)

    x = cnn_to_sequence(x, max_sequence_length, 128)

    input_length = x.shape[1]
    # start at 4096

    while input_length > max_sequence_length:
        x = create_attention_module(x, 8, 1)
        #      # Apply Conv1D with calculated strides and kernel size
        x = layers.Conv1D(filters=num_chars, kernel_size=2,
                          strides=2, padding='valid')(x)

        input_length = x.shape[1]

    x = create_attention_module(x, 8, 4)

    x = create_dense_architecture(x, num_chars, 1, 0.1)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(x)

    model = Model(inputs, outputs, name='qr_model')

    return model
