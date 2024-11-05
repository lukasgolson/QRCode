import keras
import tensorflow as tf
from keras import Layer, Sequential, Input, Model
from keras.src.layers import Conv2D, Flatten, Dense, Reshape, LeakyReLU, MaxPooling2D, BatchNormalization, Dropout, \
    SpatialDropout2D, Add, Concatenate


class SpatialTransformer(Layer):
    def __init__(self, output_intermediaries=True, **tfwargs):
        super(SpatialTransformer, self).__init__(**tfwargs)
        # Define the localization networtf parameters

        self.output_intermediaries = output_intermediaries
        self.localization_network = None
        self.trans_param_network = None

        self.concatenate_layer = None

    def create_localization_network(self, input_shape):
        # Define the input layer
        inputs = Input(shape=input_shape)

        # Convolutional layers for feature extraction with batch normalization and dropout
        x = Conv2D(6, (3, 3), padding='same', activation='relu')(inputs)
        x = MaxPooling2D()(x)

        x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(6, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D()(x)

        outputs = x

        # Create the model
        return Model(inputs=inputs, outputs=outputs)

    def create_transformation_paramater_network(self, input_shape):
        inputs = Input(shape=input_shape)

        # Flatten and Dense layers for final transformation parameters
        x = Flatten()(inputs)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(96, activation='relu')(x)
        x = Dropout(0.2)(x)

        # Output layer initialized to identity transformation
        outputs = Dense(6, activation='linear', kernel_initializer='zeros',
                        bias_initializer=tf.constant_initializer([1, 0, 0, 0, 1, 0]))(x)

        return Model(inputs=inputs, outputs=outputs)

    def build(self, input_shape):
        self.input_shape = input_shape

        self.localization_network = self.create_localization_network(input_shape[1:])

        self.localization_network.build(input_shape)

        localization_network_output_shape = self.localization_network.compute_output_shape(input_shape)

        self.trans_param_network = self.create_transformation_paramater_network(localization_network_output_shape[1:])

        self.trans_param_network.build(localization_network_output_shape)

        if self.output_intermediaries:
            self.concatenate_layer = Concatenate(axis=-1)

            # Determine the number of channels from the localization network output
            localized_channels = localization_network_output_shape[-1]

            # Define the shape for the resized localized output
            # Make sure to match the height and width of the original input shape
            resized_localized_shape = (input_shape[0], input_shape[1], input_shape[2], localized_channels)

            # Build the concatenate layer with the correct input shapes
            self.concatenate_layer.build([input_shape, resized_localized_shape])


        super(SpatialTransformer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.output_intermediaries:
            return self.concatenate_layer.compute_output_shape(
                [input_shape, self.localization_network.compute_output_shape(input_shape)])
        else:
            return input_shape  # Output shape is the same as input image shape

    @tf.function
    def call(self, inputs):
        # Unpack the inputs
        x = inputs  # x is the input image/feature map

        # Reduce channels by taking the average across the channel dimension
        # x = tf.reduce_mean(x, axis=-1, keepdims=True)  # Shape: (batch_size, height, width)

        theta, x_transformed = self.localization_network(x)

        # Generate a grid of coordinates
        grid = self._generate_grid(theta, self.input_shape[0:3])  # Get height and width from input shape

        if self.output_intermediaries:
            resized = tf.image.resize(x_transformed, (self.input_shape[1], self.input_shape[2]), method='nearest')
            x = self.concatenate_layer([x, resized])

        # Sample the input using the generated grid
        x_transformed = self._sampler(x, grid)
        return x_transformed

    def get_config(self):
        base_config = super(SpatialTransformer, self).get_config()
        config = {
            'output_intermediaries': self.output_intermediaries
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def _generate_grid(self, theta, output_size):
        # Get height and width from output_size
        batch_size, height, width = output_size

        # Create normalized grid
        x_grid = tf.tile(tf.range(0, width, dtype=tf.float32)[None, None, :],
                         (1, height, 1))  # Shape: (1, height, width)
        y_grid = tf.tile(tf.range(0, height, dtype=tf.float32)[None, :, None],
                         (1, 1, width))  # Shape: (1, height, width)

        # Normalize the grid to the range [-1, 1]
        x_grid = (x_grid / (width - 1)) * 2 - 1
        y_grid = (y_grid / (height - 1)) * 2 - 1

        # Stack grids to create the full grid
        grid = tf.stack([x_grid, y_grid], axis=-1)  # Shape: (1, height, width, 2)

        # print(f"Initial grid shape: {grid.shape}")  # Should be (1, height, width, 2)

        grid = tf.reshape(grid, (1, height * width, 2))  # Shape: (1, height * width, 2)

        # print(f"Reshaped grid shape: {grid.shape}")  # Should be (1, height * width, 2)

        batch_size = tf.shape(theta)[0]  # Determine the batch size from theta

        grid = tf.tile(grid, [batch_size, 1, 1])  # Shape: (batch_size, height * width, 2)

        theta = tf.reshape(theta, (batch_size, 2, 3))  # Shape: (batch_size, 2, 3)

        # print(f"grid rep: {grid.shape}, theta: {theta.shape}")

        affine_matrix = theta[:, :, :2]  # Shape: (batch_size, 2, 2)
        translation = theta[:, :, 2]  # Shape: (batch_size, 2)

        transformed_grid = tf.linalg.matmul(grid, affine_matrix)  # Resulting shape: (batch_size, height * width, 2)

        #   print(f"Transformed grid shape: {transformed_grid.shape}")  # Should be (batch_size, height * width, 2)

        transformed_grid = transformed_grid + translation[:, None, :]  # Broadcast translation to all grid points

        #  print(f"Transformed grid shape (2): {transformed_grid.shape}")  # Should be (batch_size, height * width, 2)

        reshaped = tf.reshape(transformed_grid, (batch_size, height, width, 2))

        #   print(f"Reshaped transformed grid shape: {reshaped.shape}")  # Should be (batch_size, height, width, 2)

        return reshaped

    @tf.function
    def _sampler(self, img, grid):
        """
        Performs bilinear sampling of the input images according to the normalized
        coordinate grid.

        img: batch of images in shape (batch_size, height, width, channels)
        grid: transformed grid of shape (batch_size, height, width, 2)

        Returns:
        sampled_img: the transformed images
        """
        batch_size, height, width, channels = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], tf.shape(img)[3]

        # Unpack grid coordinates
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        # Scale grid from [-1, 1] to image coordinates
        x = 0.5 * ((x + 1.0) * tf.cast(width - 1, tf.float32))
        y = 0.5 * ((y + 1.0) * tf.cast(height - 1, tf.float32))

        # Get the corner pixel values around the transformed coordinates
        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))

        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # Sample the image at the corners
        img = tf.reshape(img, [batch_size, height, width, channels])
        Ia = tf.gather_nd(img, tf.stack([tf.cast(y0, tf.int32), tf.cast(x0, tf.int32)], axis=-1), batch_dims=1)
        Ib = tf.gather_nd(img, tf.stack([tf.cast(y1, tf.int32), tf.cast(x0, tf.int32)], axis=-1), batch_dims=1)
        Ic = tf.gather_nd(img, tf.stack([tf.cast(y0, tf.int32), tf.cast(x1, tf.int32)], axis=-1), batch_dims=1)
        Id = tf.gather_nd(img, tf.stack([tf.cast(y1, tf.int32), tf.cast(x1, tf.int32)], axis=-1), batch_dims=1)

        # Compute the weighted sum of the corners
        sampled_img = wa[:, :, :, None] * Ia + wb[:, :, :, None] * Ib + wc[:, :, :, None] * Ic + wd[:, :, :, None] * Id

        return sampled_img
