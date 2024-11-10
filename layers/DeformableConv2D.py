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
                dtype=self.compute_dtype,
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

    @tf.function
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
        # Define the shapes for efficient indexing
        batch_size = tf.shape(inputs)[0]
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        num_sampling_points = tf.shape(sampling_locations)[1]

        # Precast sampling locations once
        x, y = tf.cast(sampling_locations[..., 0], dtype=self.compute_dtype), tf.cast(sampling_locations[..., 1], dtype=self.compute_dtype)

        # Calculate integer floor and ceiling coordinates
        x0, y0 = tf.floor(x), tf.floor(y)
        x1, y1 = x0 + 1, y0 + 1

        # Convert width and height to self.compute_dtype for consistency
        width, height = tf.cast(width, self.compute_dtype), tf.cast(height, self.compute_dtype)

        # Clip coordinates
        x0, x1 = tf.clip_by_value(x0, 0, width - 1), tf.clip_by_value(x1, 0, width - 1)
        y0, y1 = tf.clip_by_value(y0, 0, height - 1), tf.clip_by_value(y1, 0, height - 1)

        # Bilinear interpolation weights
        wa, wb = (x1 - x) * (y1 - y), (x1 - x) * (y - y0)
        wc, wd = (x - x0) * (y1 - y), (x - x0) * (y - y0)

        # Convert indices to int32 once
        x0, x1 = tf.cast(x0, tf.int32), tf.cast(x1, tf.int32)
        y0, y1 = tf.cast(y0, tf.int32), tf.cast(y1, tf.int32)

        # Create base indices for each batch
        batch_indices = tf.range(batch_size)[:, None]
        batch_indices = tf.tile(batch_indices, [1, num_sampling_points])

        # Reshape batch_indices to match y0 and x0 shapes
        batch_indices = tf.reshape(batch_indices, [batch_size, num_sampling_points])

        # Stack indices for gather_nd (shape: [batch_size, num_sampling_points, 3])
        idx_a = tf.stack([batch_indices, y0, x0], axis=-1)
        idx_b = tf.stack([batch_indices, y1, x0], axis=-1)
        idx_c = tf.stack([batch_indices, y0, x1], axis=-1)
        idx_d = tf.stack([batch_indices, y1, x1], axis=-1)

        # Gather values using gather_nd
        Ia = tf.gather_nd(inputs, idx_a)
        Ib = tf.gather_nd(inputs, idx_b)
        Ic = tf.gather_nd(inputs, idx_c)
        Id = tf.gather_nd(inputs, idx_d)

        # Reshape and apply weights in-place to avoid redundant expansions
        wa, wb, wc, wd = map(lambda w: tf.expand_dims(w, axis=-1), [wa, wb, wc, wd])

        # Compute final interpolated values
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output




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
