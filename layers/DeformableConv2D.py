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
        self.offset_channels = 2 * kernel_size * kernel_size  # Offset channels per kernel position

        # Number of filters per group
        self.group_filters = filters // num_groups

        # Initial 1x1 convolution to adjust channel count for grouping
        self.initial_conv_layer = None

        # One offset convolution layer per group (shared across channels within each group)
        self.conv_offset_groups = [
            Conv2D(
                self.offset_channels,  # Only one set of offsets per group
                kernel_size=kernel_size,
                padding="same",
                kernel_initializer=Zeros(),
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

        self.initial_conv_layer = Conv2D(
            filters=input_shape[-1],
            kernel_size=1,
            padding="same"
        )

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
        # Initial convolution to adjust channel arrangement
        inputs = self.initial_conv_layer(inputs)

        # Split channels into groups
        group_inputs = tf.split(inputs, self.num_groups, axis=-1)
        group_outputs = []

        # Process each group sequentially with dependencies
        for i in range(self.num_groups):
            with tf.control_dependencies(group_outputs):
                # Calculate the offset for the current group
                offsets = self.conv_offset_groups[i](group_inputs[i])

                # Sample using the offset map for this group
                sampled_values = self._sample_with_offsets(group_inputs[i], offsets)

                # Apply the group convolution to the sampled values
                group_output = self.conv_groups[i](sampled_values)
                group_outputs.append(group_output)  # Store result for each group

        # Concatenate group outputs along the channel axis at the end
        outputs = tf.concat(group_outputs, axis=-1)
        return outputs


    @tf.function
    def _sample_with_offsets(self, inputs, offsets):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[3]

        # Prepare the grid
        grid_y, grid_x = tf.meshgrid(
            tf.range(height, dtype=tf.float32), tf.range(width, dtype=tf.float32), indexing="ij"
        )
        grid = tf.stack((grid_x, grid_y), axis=-1)  # Shape: [height, width, 2]
        grid = tf.expand_dims(grid, axis=0)  # Shape: [1, height, width, 2]

        # Reshape offsets to apply one shared mask per group
        offsets = tf.reshape(
            offsets, [batch_size, height, width, self.kernel_size * self.kernel_size, 2]
        )

        # Generate kernel grid offsets
        kernel_grid = self._get_kernel_grid()
        kernel_grid = tf.reshape(kernel_grid, [1, 1, 1, self.kernel_size * self.kernel_size, 2])

        # Compute sampling locations by adding the offsets
        sampling_locations = grid[:, :, :, None, :] + kernel_grid + offsets

        # Clip sampling locations to stay within bounds
        sampling_locations = tf.stack(
            [
                tf.clip_by_value(sampling_locations[..., 0], 0, tf.cast(width - 1, tf.float32)),
                tf.clip_by_value(sampling_locations[..., 1], 0, tf.cast(height - 1, tf.float32)),
            ],
            axis=-1,
        )

        # Flatten sampling locations for interpolation
        sampling_locations = tf.reshape(
            sampling_locations, [batch_size, height * width * self.kernel_size * self.kernel_size, 2]
        )

        # Perform bilinear interpolation
        sampled_values = self._bilinear_interpolate(inputs, sampling_locations)

        # Reshape sampled values to [batch_size, height, width, kernel_size * kernel_size * channels]
        sampled_values = tf.reshape(
            sampled_values,
            [batch_size, height, width, self.kernel_size * self.kernel_size * channels],
        )

        return sampled_values

    @tf.function
    def _get_kernel_grid(self):
        offset = (self.kernel_size - 1) / 2.0
        x = tf.linspace(-offset, offset, self.kernel_size)
        y = tf.linspace(-offset, offset, self.kernel_size)
        x_grid, y_grid = tf.meshgrid(x, y)
        kernel_grid = tf.stack([x_grid, y_grid], axis=-1)
        kernel_grid = tf.reshape(kernel_grid, [-1, 2])  # Shape: [kernel_size * kernel_size, 2]
        return kernel_grid

    @tf.function

    def _bilinear_interpolate(self, inputs, sampling_locations):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[3]
        num_sampling_points = tf.shape(sampling_locations)[1]

        x = sampling_locations[..., 0]
        y = sampling_locations[..., 1]

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        inputs_flat = tf.reshape(inputs, [batch_size * height * width, channels])

        base = tf.range(batch_size) * height * width
        base = tf.reshape(base, [batch_size, 1])
        base = tf.tile(base, [1, num_sampling_points])

        idx_a = base + y0 * width + x0
        idx_b = base + y1 * width + x0
        idx_c = base + y0 * width + x1
        idx_d = base + y1 * width + x1

        Ia = tf.gather(inputs_flat, tf.reshape(idx_a, [-1]))
        Ib = tf.gather(inputs_flat, tf.reshape(idx_b, [-1]))
        Ic = tf.gather(inputs_flat, tf.reshape(idx_c, [-1]))
        Id = tf.gather(inputs_flat, tf.reshape(idx_d, [-1]))

        Ia = tf.reshape(Ia, [batch_size, num_sampling_points, channels])
        Ib = tf.reshape(Ib, [batch_size, num_sampling_points, channels])
        Ic = tf.reshape(Ic, [batch_size, num_sampling_points, channels])
        Id = tf.reshape(Id, [batch_size, num_sampling_points, channels])

        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return interpolated

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
