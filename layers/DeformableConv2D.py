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

        # Generate kernel grid offsets, cached as constant
        kernel_grid = tf.reshape(self._get_kernel_grid(), [1, 1, 1, self.kernel_size * self.kernel_size, 2])

        # Create spatial grid only once
        grid_y, grid_x = tf.meshgrid(tf.range(height, dtype=tf.float32), tf.range(width, dtype=tf.float32), indexing="ij")
        grid = tf.cast(tf.stack((grid_x, grid_y), axis=-1), dtype=self.compute_dtype)  # Pre-cast grid
        grid = tf.expand_dims(grid, axis=0)

        # Reshape offsets to match kernel structure
        offsets = tf.reshape(offsets, [batch_size, height, width, self.kernel_size * self.kernel_size, 2])

        # Calculate sampling locations, avoiding intermediate tensors
        sampling_locations = tf.add(tf.add(grid[:, :, :, None, :], kernel_grid), offsets)

        # Clip sampling locations and flatten
        sampling_locations = tf.clip_by_value(
            sampling_locations, 0, tf.cast([width - 1, height - 1], self.compute_dtype)
        )
        sampling_locations = tf.reshape(sampling_locations, [batch_size, height * width * self.kernel_size * self.kernel_size, 2])

        # Interpolate and reshape
        sampled_values = self._batched_bilinear_interpolate(inputs, sampling_locations)
        return tf.reshape(sampled_values, [batch_size, height, width, self.kernel_size * self.kernel_size * channels])



    @tf.function
    def _get_kernel_grid(self):
        offset = (self.kernel_size - 1) / 2.0
        x = tf.linspace(-offset, offset, self.kernel_size)
        y = tf.linspace(-offset, offset, self.kernel_size)
        x_grid, y_grid = tf.meshgrid(x, y)
        return tf.cast(tf.reshape(tf.stack([x_grid, y_grid], axis=-1), [-1, 2]), dtype=self.compute_dtype)

    @tf.function
    def _batched_bilinear_interpolate(self, inputs, sampling_locations):
        batch_size = tf.shape(inputs)[0]
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        num_sampling_points = tf.shape(sampling_locations)[1]

        # Separate x and y, using int rounding before indexing
        x, y = tf.cast(sampling_locations[..., 0], self.compute_dtype), tf.cast(sampling_locations[..., 1], self.compute_dtype)
        x0, y0 = tf.floor(x), tf.floor(y)
        x1, y1 = x0 + 1, y0 + 1

        # Pre-cast width/height and clip coordinates
        width, height = tf.cast(width, self.compute_dtype), tf.cast(height, self.compute_dtype)
        x0, x1 = tf.clip_by_value(x0, 0, width - 1), tf.clip_by_value(x1, 0, width - 1)
        y0, y1 = tf.clip_by_value(y0, 0, height - 1), tf.clip_by_value(y1, 0, height - 1)

        # Define weights in place, reshaping to match gathered tensor channels
        wa = tf.expand_dims((x1 - x) * (y1 - y), axis=-1)
        wb = tf.expand_dims((x1 - x) * (y - y0), axis=-1)
        wc = tf.expand_dims((x - x0) * (y1 - y), axis=-1)
        wd = tf.expand_dims((x - x0) * (y - y0), axis=-1)

        # Prepare integer indices
        x0, x1, y0, y1 = map(lambda coord: tf.cast(coord, tf.int32), [x0, x1, y0, y1])
        batch_indices = tf.tile(tf.range(batch_size)[:, None], [1, num_sampling_points])

        # Gather indices and values, and reshape inputs to support broadcasting with weights
        def gather_values(y, x):
            indices = tf.stack([batch_indices, y, x], axis=-1)
            gathered_values = tf.gather_nd(inputs, indices)
            return tf.reshape(gathered_values, [batch_size, num_sampling_points, channels])

        Ia = gather_values(y0, x0)
        Ib = gather_values(y1, x0)
        Ic = gather_values(y0, x1)
        Id = gather_values(y1, x1)

        # Perform weighted sum
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
