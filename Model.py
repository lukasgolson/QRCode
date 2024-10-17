import keras
import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.src.layers import MultiHeadAttention
from tensorflow.keras.layers import Dropout

from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformerInputHead
from layers.involution import Involution


@keras.saving.register_keras_serializable(package="qr_model", name="positional_encoding")
def positional_encoding(length, depth):
    """Generates a positional encoding matrix for a given sequence length and depth (embedding size)."""
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))

    # Apply sine to even indices and cosine to odd indices
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices in the array
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices in the array

    pos_encoding = pos_encoding[np.newaxis, ...]  # Add batch dimension
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_involution_architecture(input_tensor, length, min_resolution=64, max_channels=128, group_number=1):
    x = input_tensor

    height = input_tensor.shape[1]
    width = input_tensor.shape[2]
    side_length = min(height, width)

    layer_channels = input_tensor.shape[-1] + 1  # Get the number of channels in the input tensor

    for i in range(length):

        print(f"Involution layer {i}")
        layer_channels = min(layer_channels * (2 ** i), max_channels)

        residual = x

        # Convolution to adjust the number of channels
        x = keras.layers.Conv2D(layer_channels, (1, 1), activation='mish')(x)

        # Involution layer with stride 1 (to avoid automatic downscaling)
        x, _ = Involution(
            channel=layer_channels, group_number=group_number, kernel_size=3, stride=1, reduction_ratio=2)(x)

        x = layers.BatchNormalization()(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = keras.layers.Conv2D(x.shape[-1], (1, 1))(residual)  # Adjust dimensions if needed
        x += residual  # Add the residual connectio
        x = keras.layers.ReLU()(x)  # Re-apply activation





        if side_length > min_resolution:
            x = keras.layers.Conv2D(layer_channels, (1, 1), strides=(2, 2))(x)
            side_length //= 2  # Update the current height to reflect the downscaling

        # x = SpatialAttention()(x)

    return x


def create_encoder_architecture(input_tensor, max_sequence_length=512, num_chars=128):
    x = input_tensor

    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(max_sequence_length, activation='mish')(x)
    x = layers.Reshape((max_sequence_length, -1))(x)

    pos_encoding = positional_encoding(max_sequence_length, x.shape[-1])
    x += pos_encoding

    x = MultiHeadAttention(num_heads=8, key_dim=x.shape[-1])(x, x)

    x = layers.Dense(num_chars, activation='mish')(x)

    x = layers.BatchNormalization()(x)

    print(x.shape)

    return x


def create_dense_architecture(input_tensor, units=512, min_units=96, depth=3):
    x = input_tensor

    # Initialize a list to hold the units for each layer
    units_per_layer = []

    # Calculate the total distribution while keeping it pyramidal
    remaining_units = units

    for i in range(depth):
        # Calculate the units for the current layer
        if i < depth - 1:
            # Distribute remaining units evenly but ensure no layer goes below min_units
            current_units = max(min_units, remaining_units // (depth - i))
        else:
            # Assign the rest to the last layer, ensuring it's at least min_units
            current_units = max(min_units, remaining_units)

        units_per_layer.append(current_units)
        remaining_units -= current_units

    # Build the model using the calculated units with residual layers
    for i, current_units in enumerate(units_per_layer):
        # Save the input to add it later for the residual connection
        residual = x

        # Main branch
        x = layers.Dense(current_units, activation='mish')(x)
        x = Dropout(0.25)(x)
        x = layers.BatchNormalization()(x)

        # Projection if necessary to match dimensions
        if residual.shape[-1] != current_units:
            residual = layers.Dense(current_units)(residual)  # Match dimensions

        # Add the residual connection
        x = layers.Add()([x, residual])

        x = layers.ReLU()(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Instantiate the SpatialTransformerInputHead
    processing_head = SpatialTransformerInputHead()(inputs)  # Ensure the output is used correctly

    # Build the involution architecture

    x = SpatialAttention()(processing_head)

    x = create_involution_architecture(x, 3, 16, 4)

    x = create_encoder_architecture(x, max_sequence_length)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)

    x = create_dense_architecture(x, 512, 98,3)

    x = layers.Dense(num_chars, activation='mish')(x)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(x)

    return Model(inputs, outputs, name='qr_model')
