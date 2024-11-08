import tensorflow as tf
from keras.src.layers import Conv3D
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.initializers import Zeros


class DeformableConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.offset_channels = (
                2 * kernel_size * kernel_size
        )  # Offset channels for x and y directions
        self.conv_offset = Conv2D(
            self.offset_channels,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=Zeros(),
        )
        self.conv = Conv2D(filters, kernel_size=1, padding="same")

    def build(self, input_shape):
        self.conv_offset.build(input_shape)
        num_offsets = self.kernel_size * self.kernel_size
        channels = input_shape[-1] * num_offsets
        conv_input_shape = input_shape[:-1] + (channels,)
        self.conv.build(conv_input_shape)
        self.built = True
        _, self.height, self.width, _ = input_shape

    def call(self, inputs):
        # Generate offsets
        offsets = self.conv_offset(inputs)

        # Get input shape and kernel dimensions
        batch_size = tf.shape(inputs)[0]
        height = self.height
        width = self.width

        # Create a mesh grid for standard convolutional positions
        y, x = tf.meshgrid(
            tf.range(height, dtype=tf.float32),
            tf.range(width, dtype=tf.float32),
            indexing="ij",
        )
        # Expand dims to make x and y broadcastable
        x = tf.expand_dims(x, axis=0)    # Shape: [1, height, width]
        x = tf.expand_dims(x, axis=-1)   # Shape: [1, height, width, 1]
        y = tf.expand_dims(y, axis=0)
        y = tf.expand_dims(y, axis=-1)

        # Apply offsets to each kernel point
        x_offsets = offsets[..., ::2]
        y_offsets = offsets[..., 1::2]

        # New sampling locations
        new_x = x + x_offsets
        new_y = y + y_offsets

        # Clip values to stay within image boundaries
        new_x = tf.clip_by_value(new_x, 0, width - 1)
        new_y = tf.clip_by_value(new_y, 0, height - 1)

        # Interpolation (bilinear sampling)
        sampled_points = self.bilinear_interpolate(inputs, new_x, new_y)
        # Reshape sampled_points to combine num_offsets and channels dimensions
        batch_size = tf.shape(sampled_points)[0]
        height = tf.shape(sampled_points)[1]
        width = tf.shape(sampled_points)[2]
        num_offsets = tf.shape(sampled_points)[3]
        channels = tf.shape(sampled_points)[4]
        sampled_points = tf.reshape(
            sampled_points, [batch_size, height, width, num_offsets * channels]
        )

        # Apply convolution on the reshaped sampled points
        outputs = self.conv(sampled_points)
        return outputs


    def bilinear_interpolate(self, inputs, x, y):
        # Get shapes
        batch_size = tf.shape(inputs)[0]
        height = self.height
        width = self.width

        # Number of offsets
        num_offsets = x.shape[-1]

        # Perform bilinear interpolation
        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        # Clip and cast values
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        # Generate batch indices
        batch_indices = tf.range(batch_size)
        batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1, 1])
        batch_indices = tf.broadcast_to(batch_indices, tf.shape(x0))

        # Stack indices
        indices_a = tf.stack([batch_indices, y0, x0], axis=-1)
        indices_b = tf.stack([batch_indices, y1, x0], axis=-1)
        indices_c = tf.stack([batch_indices, y0, x1], axis=-1)
        indices_d = tf.stack([batch_indices, y1, x1], axis=-1)

        # Gather pixel values
        Ia = tf.gather_nd(inputs, indices_a)
        Ib = tf.gather_nd(inputs, indices_b)
        Ic = tf.gather_nd(inputs, indices_c)
        Id = tf.gather_nd(inputs, indices_d)

        # Calculate interpolation weights
        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = (x1_f - x) * (y1_f - y)
        wb = (x1_f - x) * (y - y0_f)
        wc = (x - x0_f) * (y1_f - y)
        wd = (x - x0_f) * (y - y0_f)

        # Expand dims to match shapes
        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        # Interpolate
        interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return interpolated

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.filters,)

    def get_config(self):
        config = super(DeformableConv2D, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config
