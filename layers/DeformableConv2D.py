import tensorflow as tf
from keras.src.layers import DepthwiseConv2D
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.initializers import Zeros

class DeformableConv2D(Layer):
    def __init__(self, filters, kernel_size, num_groups=1, **kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.offset_channels = 2 * kernel_size * kernel_size
        self.group_filters = filters // num_groups

        # Offset convolution layers for each group
        self.conv_offset_groups = [
            Conv2D(
                self.offset_channels,
                kernel_size=kernel_size,
                padding="same",
                kernel_initializer=Zeros(),
                dtype="float16",  # Set float16 precision for offsets
            ) for _ in range(num_groups)
        ]

        # Convolutional layers for each group
        self.conv_groups = [
            tf.keras.Sequential([
                DepthwiseConv2D(kernel_size=self.kernel_size, padding="same"),
                Conv2D(self.group_filters, kernel_size=1, padding="same")
            ])
            for _ in range(num_groups)
        ]

    def build(self, input_shape):
        batch_size, height, width, channels = input_shape
        assert channels % self.num_groups == 0, "Channels must be divisible by num_groups"

        self.group_channels = channels // self.num_groups

        # Build offset and convolution layers for each group
        for conv_offset in self.conv_offset_groups:
            conv_offset.build([batch_size, height, width, self.group_channels])

        for conv in self.conv_groups:
            conv.build([batch_size, height, width, self.kernel_size * self.kernel_size * self.group_channels])

        super(DeformableConv2D, self).build(input_shape)

    def call(self, inputs):
        group_inputs = tf.split(inputs, self.num_groups, axis=-1)
        group_outputs = []

        # Process each group independently
        for i in range(self.num_groups):
            offsets = tf.cast(self.conv_offset_groups[i](group_inputs[i]), dtype=self.compute_dtype)  # Ensure offsets are float16
            sampled_values = self._sample_with_offsets(tf.cast(group_inputs[i], dtype=self.compute_dtype), offsets)
            group_output = self.conv_groups[i](sampled_values)
            group_outputs.append(group_output)

        # Concatenate group outputs
        outputs = tf.concat(group_outputs, axis=-1)
        return outputs

    @tf.function
    def _sample_with_offsets(self, inputs, offsets):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[3]

        # Generate kernel grid offsets in float16
        kernel_grid = tf.cast(self._get_kernel_grid(), dtype=self.compute_dtype)
        kernel_grid = tf.reshape(kernel_grid, [1, 1, 1, self.kernel_size * self.kernel_size, 2])

        # Prepare the grid (using float32 for tf.range and casting later)
        grid_y, grid_x = tf.meshgrid(
            tf.range(height, dtype=tf.float32), tf.range(width, dtype=tf.float32), indexing="ij"
        )
        grid = tf.stack((grid_x, grid_y), axis=-1)
        grid = tf.cast(grid, dtype=self.compute_dtype)  # Cast the grid to float16 here
        grid = tf.expand_dims(grid, axis=0)

        # Reshape offsets
        offsets = tf.reshape(offsets, [batch_size, height, width, self.kernel_size * self.kernel_size, 2])

        # Compute sampling locations
        sampling_locations = grid[:, :, :, None, :] + kernel_grid + offsets

        # Clip sampling locations
        sampling_locations = tf.stack([
            tf.clip_by_value(sampling_locations[..., 0], 0, tf.cast(width - 1, self.compute_dtype)),
            tf.clip_by_value(sampling_locations[..., 1], 0, tf.cast(height - 1, self.compute_dtype))
        ], axis=-1)

        # Flatten sampling locations for interpolation
        sampling_locations = tf.reshape(
            sampling_locations, [batch_size, height * width * self.kernel_size * self.kernel_size, 2]
        )

        # Perform interpolation with float16 precision
        sampled_values = self._batched_bilinear_interpolate(tf.cast(inputs, dtype=self.compute_dtype), sampling_locations)

        # Reshape sampled values
        return tf.reshape(
            sampled_values, [batch_size, height, width, self.kernel_size * self.kernel_size * channels]
        )


    @tf.function
    def _get_kernel_grid(self):
        offset = (self.kernel_size - 1) / 2.0
        x = tf.linspace(-offset, offset, self.kernel_size)
        y = tf.linspace(-offset, offset, self.kernel_size)
        x_grid, y_grid = tf.meshgrid(x, y)
        return tf.cast(tf.reshape(tf.stack([x_grid, y_grid], axis=-1), [-1, 2]), dtype=self.compute_dtype)

    @tf.function
    def _batched_bilinear_interpolate(self, inputs, sampling_locations):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[3]
        num_sampling_points = tf.shape(sampling_locations)[1]

        x = tf.cast(sampling_locations[..., 0], dtype=self.compute_dtype)
        y = tf.cast(sampling_locations[..., 1], dtype=self.compute_dtype)

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        # Clip coordinates
        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, self.compute_dtype))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, self.compute_dtype))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, self.compute_dtype))
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, self.compute_dtype))

        # Bilinear interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # Convert indices to int32 for gathering
        x0, x1 = tf.cast(x0, tf.int32), tf.cast(x1, tf.int32)
        y0, y1 = tf.cast(y0, tf.int32), tf.cast(y1, tf.int32)

        # Flatten inputs for gather
        inputs_flat = tf.reshape(inputs, [batch_size * height * width, channels])
        base = tf.range(batch_size) * height * width
        base = tf.reshape(base, [batch_size, 1])
        base = tf.tile(base, [1, num_sampling_points])

        # Compute indices for each corner
        idx_a = base + y0 * width + x0
        idx_b = base + y1 * width + x0
        idx_c = base + y0 * width + x1
        idx_d = base + y1 * width + x1

        # Gather and apply weights
        Ia = tf.gather(inputs_flat, tf.reshape(idx_a, [-1]))
        Ib = tf.gather(inputs_flat, tf.reshape(idx_b, [-1]))
        Ic = tf.gather(inputs_flat, tf.reshape(idx_c, [-1]))
        Id = tf.gather(inputs_flat, tf.reshape(idx_d, [-1]))

        # Reshape and apply weights
        Ia = tf.reshape(Ia, [batch_size, num_sampling_points, channels])
        Ib = tf.reshape(Ib, [batch_size, num_sampling_points, channels])
        Ic = tf.reshape(Ic, [batch_size, num_sampling_points, channels])
        Id = tf.reshape(Id, [batch_size, num_sampling_points, channels])

        wa, wb, wc, wd = map(lambda w: tf.cast(tf.expand_dims(w, axis=-1), dtype=self.compute_dtype), [wa, wb, wc, wd])
        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.filters,)

    def get_config(self):
        config = super(DeformableConv2D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "num_groups": self.num_groups,
        })
        return config
