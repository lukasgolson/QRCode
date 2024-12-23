from keras import layers, Model
from keras.src.layers import MultiHeadAttention, Conv2D, Add
from tensorflow.keras.layers import Dropout, BatchNormalization

from layers.ExtractPatches import ExtractPatches
from layers.PositionalEncoding import PositionalEncoding
from layers.SpatialTransformer import SpatialTransformer

activation = "relu"


def calculate_downscale_frequency(length, min_resolution, max_resolution):
    # Calculate total downscales possible
    total_downscales = 0
    current_height = max_resolution
    current_width = max_resolution

    while current_height > min_resolution and current_width > min_resolution:
        current_height //= 2
        current_width //= 2

        if current_height >= min_resolution and current_width >= min_resolution:
            total_downscales += 1

    # Calculate downscale frequency
    if total_downscales > 0:
        downscale_frequency = length // total_downscales
    else:
        downscale_frequency = 0  # If no downscale possible, set to 0

    return downscale_frequency


def create_cnn_architecture(input_tensor, length, min_resolution=64, max_channels=64, attention_frequency=2,
                            use_residual=True):
    x = input_tensor
    current_height = input_tensor.shape[1]  # Current height of the input image

    # Calculate downscale frequency
    downscale_frequency = calculate_downscale_frequency(length, min_resolution, current_height)

    x = layers.Conv2D(3, (1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)

    current_channels = x.shape[-1]

    for i in range(length):
        if i % attention_frequency == 0:
            # x = SqueezeExcitation(use_residual=True)(x)
            # x = SpatialAttention(use_residual=True)(x)
            x = x

        # Calculate the current number of channels

        residual = x

        x = Conv2D(current_channels, (3, 3), padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = Conv2D(current_channels, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)

        if use_residual:
            # Ensure the residual has the same number of channels
            if residual.shape[-1] != x.shape[-1]:
                residual = Conv2D(x.shape[-1], (1, 1))(residual)

            x = Add()([x, residual])

            x = layers.LeakyReLU()(x)

            x = BatchNormalization()(x)

        if downscale_frequency > 0 and i % downscale_frequency == 0 and i > 0:
            x = Conv2D(current_channels, (1, 1), strides=(2, 2), padding='same')(x)
            x = layers.LeakyReLU()(x)

        current_channels = min(current_channels * (2 ** i), max_channels)

    return x


def create_dense_architecture(input_tensor, units=512, depth=3, dropout=0.1):
    x = input_tensor

    # Build the model using the same number of units for each layer
    for i in range(depth):
        # Save the input to add as a residual if necessary
        residual = x

        # Apply dense layer with a constant number of units
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)

        x = Dropout(dropout)(x)

        # Add residual connection after the first layer
        if i > 0:
            x = layers.Add()([x, residual])  # Residual connection

        # Apply normalization and activation after the add operation
            x = layers.LeakyReLU()(x)


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

        x = layers.Dense(x.shape[-1])(x)
        x = layers.LeakyReLU()(x)

        x = Dropout(dropout)(x)
        x = layers.Dense(x.shape[-1])(x)


        x = layers.Add()([x, residual_2])
        x = layers.LeakyReLU()(x)


        x = layers.BatchNormalization()(x)

    return x


# The idea is to turn individual pixels into a sequence of embeddings
def cnn_to_sequence(input_tensor, max_sequence_length=512, feature_length=128, downsampled_size=64, use_cnn=False):
    x = layers.BatchNormalization()(input_tensor)

    # eventually we might want to change this to patch extraction

    x = layers.Conv2D(x.shape[-1], (1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)


    if use_cnn:
        x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

        x = layers.Conv1D(filters=feature_length, kernel_size=1, padding='valid')(x)
    else:
        # get x size and y size
        x_size = x.shape[1]
        y_size = x.shape[2]

        size = max(x_size, y_size)

        # get the stride size necessary to reduce to 64X64
        stride_size = size // downsampled_size

        # if stride size is 0, set patch and stride to not downsample
        if stride_size < 1:
            stride_size = 1

        x = ExtractPatches(stride_size=(stride_size, stride_size), patch_size=(stride_size, stride_size))(x)

        # reshape from 64, 64, 256 to 64x64, 256

        x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

    input_length = x.shape[1]
    while input_length > max_sequence_length:
        #      # Apply Conv1D with calculated strides and kernel size
        x = layers.Conv1D(filters=feature_length, kernel_size=2,
                          strides=2, padding='valid')(x)

        input_length = x.shape[1]

    x = layers.TimeDistributed(layers.Dense(feature_length))(x)
    x = layers.LeakyReLU()(x)

    x = PositionalEncoding()(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    spatial_transformer = SpatialTransformer()(inputs)

    x = create_cnn_architecture(spatial_transformer, 8, 128, 64)

    x = cnn_to_sequence(x, max_sequence_length, 128, 64)

    x = create_attention_module(x, 4, 8)

    # x = create_dense_architecture(x, num_chars, 1, 0.1)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(x)

    model = Model(inputs, outputs, name='qr_model')

    return model
