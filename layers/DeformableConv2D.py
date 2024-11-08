import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.initializers import Zeros

class DeformableConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.offset_channels = 2 * kernel_size * kernel_size  # For x and y offsets
        self.conv_offset = Conv2D(
            self.offset_channels,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=Zeros(),
        )
        self.conv = Conv2D(filters, kernel_size=kernel_size, padding="same")

    def build(self, input_shape):
        # Input shape should be [batch_size, height, width, channels]
        batch_size, height, width, channels = input_shape

        # Build the conv_offset layer (predict offsets)
        self.conv_offset.build(input_shape)  # Initialize offset prediction convolution

        # The output shape of conv_offset will have the shape [batch_size, height, width, offset_channels]
        offset_shape = (input_shape[0], height, width, self.offset_channels)

        sampled_input_shape = (batch_size, height, width, self.kernel_size * self.kernel_size * channels)
        self.conv.build(sampled_input_shape)  # Initialize the main convolution


    # Call the parent class build method to register weights
        super(DeformableConv2D, self).build(input_shape)

    def call(self, inputs):
        # Generate offsets
        offsets = self.conv_offset(inputs)  # Shape: [batch_size, height, width, offset_channels]

        # Get input shape
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            inputs.shape[-1],
        )

        # Prepare the mesh grid
        grid_y, grid_x = tf.meshgrid(
            tf.range(height, dtype=tf.float32), tf.range(width, dtype=tf.float32), indexing="ij"
        )

        # Stack and expand dimensions to create a grid
        grid = tf.stack((grid_x, grid_y), axis=-1)  # Shape: [height, width, 2]
        grid = tf.expand_dims(grid, axis=0)  # Shape: [1, height, width, 2]

        # Apply offsets
        offsets = tf.reshape(
            offsets, [batch_size, height, width, self.kernel_size * self.kernel_size, 2]
        )

        # Generate kernel grid offsets
        kernel_grid = self._get_kernel_grid()  # Shape: [kernel_size * kernel_size, 2]
        kernel_grid = tf.reshape(kernel_grid, [1, 1, 1, self.kernel_size * self.kernel_size, 2])

        # Compute the sampling locations
        sampling_locations = grid[:, :, :, None, :] + kernel_grid + offsets  # Broadcasting

        # Normalize coordinates to [0, height/width]
        sampling_locations = tf.stack(
            [
                tf.clip_by_value(sampling_locations[..., 0], 0, tf.cast(width - 1, tf.float32)),
                tf.clip_by_value(sampling_locations[..., 1], 0, tf.cast(height - 1, tf.float32)),
            ],
            axis=-1,
        )

        # Reshape for interpolation
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




        # Apply convolution
        outputs = self.conv(sampled_values)
        return outputs

    def _get_kernel_grid(self):
        # Generate relative coordinates for the kernel grid
        offset = (self.kernel_size - 1) / 2.0
        x = tf.linspace(-offset, offset, self.kernel_size)
        y = tf.linspace(-offset, offset, self.kernel_size)
        x_grid, y_grid = tf.meshgrid(x, y)
        kernel_grid = tf.stack([x_grid, y_grid], axis=-1)
        kernel_grid = tf.reshape(kernel_grid, [-1, 2])  # Shape: [kernel_size * kernel_size, 2]
        return kernel_grid

    def _bilinear_interpolate(self, inputs, sampling_locations):
        # Get shapes
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[3]
        num_sampling_points = tf.shape(sampling_locations)[1]

        # Split sampling locations
        x = sampling_locations[..., 0]
        y = sampling_locations[..., 1]

        # Get integer and fractional parts
        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        # Clip values
        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))

        # Compute interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # Expand dimensions for gathering
        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        # Flatten inputs
        inputs_flat = tf.reshape(inputs, [batch_size * height * width, channels])

        # Compute base indices
        base = tf.range(batch_size) * height * width
        base = tf.reshape(base, [batch_size, 1])
        base = tf.tile(base, [1, num_sampling_points])

        # Compute indices for each corner
        idx_a = base + y0 * width + x0
        idx_b = base + y1 * width + x0
        idx_c = base + y0 * width + x1
        idx_d = base + y1 * width + x1

        # Gather pixel values
        Ia = tf.gather(inputs_flat, tf.reshape(idx_a, [-1]))
        Ib = tf.gather(inputs_flat, tf.reshape(idx_b, [-1]))
        Ic = tf.gather(inputs_flat, tf.reshape(idx_c, [-1]))
        Id = tf.gather(inputs_flat, tf.reshape(idx_d, [-1]))

        # Reshape back to [batch_size, num_sampling_points, channels]
        Ia = tf.reshape(Ia, [batch_size, num_sampling_points, channels])
        Ib = tf.reshape(Ib, [batch_size, num_sampling_points, channels])
        Ic = tf.reshape(Ic, [batch_size, num_sampling_points, channels])
        Id = tf.reshape(Id, [batch_size, num_sampling_points, channels])

        # Compute interpolated values
        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id  # Shape: [batch_size, num_sampling_points, channels]
        return interpolated

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.filters,)

    def get_config(self):
        config = super(DeformableConv2D, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config
