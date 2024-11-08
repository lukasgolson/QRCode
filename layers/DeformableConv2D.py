import tensorflow as tf
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
        conv_input_shape = input_shape[:-1] + [channels]  # Corrected line
        self.conv.build(conv_input_shape)
        self.built = True


    def call(self, inputs):
        # Generate offsets
        offsets = self.conv_offset(inputs)  # Shape: [batch_size, height, width, 2 * num_offsets]

        # Get input shape and kernel dimensions
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = inputs.shape[-1]  # Assuming channels is known statically

        num_offsets = self.kernel_size * self.kernel_size

        # Prepare the sampling grid
        y, x = tf.meshgrid(
            tf.range(height, dtype=tf.float32),
            tf.range(width, dtype=tf.float32),
            indexing='ij'
        )  # y and x are [height, width]

        x = tf.expand_dims(x, axis=0)  # Shape: [1, height, width]
        y = tf.expand_dims(y, axis=0)  # Shape: [1, height, width]

        # Expand to match batch size
        x = tf.tile(x, [batch_size, 1, 1])  # [batch_size, height, width]
        y = tf.tile(y, [batch_size, 1, 1])  # [batch_size, height, width]

        # Reshape offsets
        offsets = tf.reshape(offsets, [batch_size, height, width, num_offsets, 2])

        # Compute sampling locations
        x_offsets = offsets[..., 0]  # [batch_size, height, width, num_offsets]
        y_offsets = offsets[..., 1]  # [batch_size, height, width, num_offsets]

        x = tf.expand_dims(x, axis=-1)  # [batch_size, height, width, 1]
        y = tf.expand_dims(y, axis=-1)  # [batch_size, height, width, 1]

        new_x = x + x_offsets  # [batch_size, height, width, num_offsets]
        new_y = y + y_offsets  # [batch_size, height, width, num_offsets]

        # Clip values to stay within image boundaries
        new_x = tf.clip_by_value(new_x, 0.0, tf.cast(width - 1, tf.float32))
        new_y = tf.clip_by_value(new_y, 0.0, tf.cast(height - 1, tf.float32))

        # Perform bilinear interpolation efficiently
        interpolated = self.bilinear_interpolate(inputs, new_x, new_y)

        # Reshape sampled points
        batch_size = tf.shape(interpolated)[0]
        height = tf.shape(interpolated)[1]
        width = tf.shape(interpolated)[2]
        num_offsets = tf.shape(interpolated)[3]
        channels = tf.shape(interpolated)[4]

        sampled_points = tf.reshape(interpolated, [batch_size, height, width, num_offsets * channels])

        # Apply 1x1 convolution
        outputs = self.conv(sampled_points)
        return outputs

    def bilinear_interpolate(self, inputs, x, y):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        num_offsets = tf.shape(x)[-1]

        # Four neighboring pixel indices
        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        # Clip indices
        x0 = tf.clip_by_value(x0, 0.0, tf.cast(width - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0.0, tf.cast(width - 1, tf.float32))
        y0 = tf.clip_by_value(y0, 0.0, tf.cast(height - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0.0, tf.cast(height - 1, tf.float32))

        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # Gather pixel values
        Ia = self.gather_values(inputs, x0, y0)
        Ib = self.gather_values(inputs, x0, y1)
        Ic = self.gather_values(inputs, x1, y0)
        Id = self.gather_values(inputs, x1, y1)

        # Expand weights
        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        # Compute interpolated values
        interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id  # Shape: [batch_size, height, width, num_offsets, channels]
        return interpolated

    def gather_values(self, inputs, x_indices, y_indices):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        num_offsets = tf.shape(x_indices)[-1]

        x_indices = tf.cast(x_indices, tf.int32)
        y_indices = tf.cast(y_indices, tf.int32)

        # Compute linear indices
        batch_indices = tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, height, width, num_offsets])

        # Flatten indices
        flat_indices = batch_indices * (height * width) + y_indices * width + x_indices  # Shape: [batch_size, height, width, num_offsets]
        flat_indices = tf.reshape(flat_indices, [-1])  # Flattened

        # Flatten inputs
        inputs_flat = tf.reshape(inputs, [-1, channels])  # Shape: [batch_size * height * width, channels]

        # Gather values
        gathered = tf.gather(inputs_flat, flat_indices)  # Shape: [batch_size * height * width * num_offsets, channels]

        # Reshape back
        gathered = tf.reshape(gathered, [batch_size, height, width, num_offsets, channels])
        return gathered

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.filters,)

    def get_config(self):
        config = super(DeformableConv2D, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config
