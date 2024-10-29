import keras
import numpy as np
import tensorflow as tf


@keras.saving.register_keras_serializable(package="qr_model", name="PositionalEncoding")
class PositionalEncoding(keras.layers.Layer):
    """Generates and adds a positional encoding matrix (signal) for a given sequence length and depth (embedding size)."""

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = None
        self.depth = None
        self.length = None

    def build(self, input_shape):
        """Build the layer and compute length and depth based on input shape."""
        # Get length and depth from the input shape
        self.length = input_shape[1]  # Sequence length (number of timesteps)
        self.depth = input_shape[2]  # Depth (embedding size)

        # Initialize the positional encoding


        # get the dtype of the input tensor
        dtype = input_shape[1].dtype
        self.pos_encoding = positional_encoding(self.length, self.depth, dtype)

    def call(self, inputs):
        """Applies the positional encoding to the input tensor."""
        # Add positional encoding to the input tensor
        return inputs + self.pos_encoding


@tf.function
@keras.saving.register_keras_serializable(package="qr_model", name="positional_encoding")
def positional_encoding(length, depth, dtype = tf.float32):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))

    # Apply sine to even indices and cosine to odd indices
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices in the array
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices in the array

    pos_encoding = pos_encoding[np.newaxis, ...]  # Add batch dimension


    # get the dtype of the input tensor


    return tf.cast(pos_encoding, dtype=dtype)
