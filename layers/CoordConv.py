import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class CoordConv(Layer):
    def __init__(self, filters, kernel_size, padding='same', **kwargs):
        super(CoordConv, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = Conv2D(filters, kernel_size=kernel_size, padding=padding)

    def build(self, input_shape):
        input_channels = input_shape[-1]

        coord_input_channels = input_channels + 2

        self.conv.build([None, input_shape[1], input_shape[2], coord_input_channels])

        super(CoordConv, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Get dynamic input shape
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Create coordinate tensors
        x_coords = tf.linspace(0.0, 1.0, width)
        y_coords = tf.linspace(0.0, 1.0, height)
        x_coords, y_coords = tf.meshgrid(x_coords, y_coords)

        # Expand dims to match input
        x_coords = tf.expand_dims(x_coords, axis=-1)  # Shape: [height, width, 1]
        y_coords = tf.expand_dims(y_coords, axis=-1)  # Shape: [height, width, 1]

        # Concatenate coordinate channels
        coord_channels = tf.concat([x_coords, y_coords], axis=-1)  # Shape: [height, width, 2]

        # Expand dims to include batch size
        coord_channels = tf.expand_dims(coord_channels, axis=0)  # Shape: [1, height, width, 2]
        coord_channels = tf.tile(coord_channels, [batch_size, 1, 1, 1])  # Shape: [batch_size, height, width, 2]

        # cast coord_channels to the same dtype as compute_dtype
        coord_channels = tf.cast(coord_channels, dtype=self.compute_dtype)

        # Concatenate with the original inputs
        concat_inputs = tf.concat([inputs, coord_channels], axis=-1)

        # Apply convolution
        return self.conv(concat_inputs)

    def get_config(self):
        config = super(CoordConv, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "padding": self.padding})
        return config
