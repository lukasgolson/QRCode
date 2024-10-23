import numpy as np
from keras import layers, Model
from keras.src.initializers import Constant
from keras.src.layers import MultiHeadAttention, Conv2D, Add, Conv3D, Conv1D
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.regularizers import l2

from layers.Activations import Mish
from layers.BilinearInterpolation import BilinearInterpolation
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

        x = Mish()(x)
        x = BatchNormalization()(x)

        x = SpatialAttention(use_skip_connection=True)(x)

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
        x = layers.Dense(units, activation='mish')(x)
        x = Dropout(dropout)(x)

        # Add residual connection after the first layer
        if i > 0:
            x = layers.Add()([x, residual])  # Residual connection

        # Apply normalization and activation after the add operation
        x = layers.Activation('mish')(x)  # 'mish' is applied here

        x = layers.BatchNormalization()(x)

    return x


def create_attention_architecture(input_tensor, heads=8, depth=3, dropout=0.1):
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

        x = layers.Dense(x.shape[-1], activation='mish')(x)
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

    x = layers.Conv2D(x.shape[-1], (1, 1), padding='same', activation='mish')(x)

    x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

    x = layers.Conv1D(filters=feature_length, kernel_size=1, padding='valid')(x)

    # Get initial input length

    input_length = x.shape[1]
    # start at 4096

    while input_length > max_sequence_length:
        # Apply Conv1D with calculated strides and kernel size
        x = layers.Conv1D(filters=feature_length, kernel_size=2,
                          strides=2, padding='valid')(x)

        input_length = x.shape[1]



    x = layers.TimeDistributed(layers.Dense(feature_length * 2, activation="mish"))(x)
    x = layers.TimeDistributed(layers.Dense(feature_length, activation="mish"))(x)

    x = PositionalEncoding()(x)

    return x


def create_input_head(inputs, out_shape=(256, 256)):
    def get_initial_transform_weights(output_size):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        w = np.zeros((output_size, 6), dtype='float32')
        return Constant(w), Constant(b.flatten())

    image = inputs
    x = layers.MaxPool2D(pool_size=(2, 2))(image)
    x = layers.Conv2D(20, (5, 5), padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(20, (5, 5), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50)(x)
    x = layers.Activation('relu')(x)

    # Use the weights from the get_initial_transform_weights
    kernel_initializer, bias_initializer = get_initial_transform_weights(50)
    x = layers.Dense(6, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)

    # Ensure BilinearInterpolation is defined or imported correctly
    interpolated_input = BilinearInterpolation(out_shape)([image, x])

    return interpolated_input


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    x = create_input_head(inputs)

    x = SpatialAttention(use_skip_connection=True)(x)

    x = create_cnn_architecture(x, 4, 64, 64)

    x = cnn_to_sequence(x, max_sequence_length, 256)

    x = create_attention_architecture(x, 8, 6)

    x = create_dense_architecture(x, num_chars, 2, 0.1)

    print(x.shape)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(x)

    model = Model(inputs, outputs, name='qr_model')

    l2_value = 0.001
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.kernel_regularizer = l2(l2_value)
        if isinstance(layer, Conv1D):
            layer.kernel_regularizer = l2(l2_value)
        if isinstance(layer, Conv2D):
            layer.kernel_regularizer = l2(l2_value)
        if isinstance(layer, Conv3D):
            layer.kernel_regularizer = l2(l2_value)
        if isinstance(layer, MultiHeadAttention):
            layer.kernel_regularizer = l2(l2_value)


    return model
