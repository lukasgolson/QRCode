import keras
from keras import layers
import tensorflow as tf
from keras.src.initializers import Constant


@keras.saving.register_keras_serializable(package="qr_model", name="SpatialTransformerInputHead")
class SpatialTransformerInputHead(keras.layers.Layer):
    def __init__(self, downscaling=2, **kwargs):
        super(SpatialTransformerInputHead, self).__init__(**kwargs)
        self.downscaling = downscaling

        # Downscaling layer for input reduction
        self.avg_pool = layers.AveragePooling2D(pool_size=(self.downscaling, self.downscaling), padding='valid')

        # Transformation model (Predicts affine transformation parameters)
        bias_initializer = Constant([1, 0, 0, 0, 1, 0])  # Initializing to identity transform
        self.transformation_model = keras.Sequential([
            layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(6, activation='linear', kernel_initializer='zeros', bias_initializer=bias_initializer)
        ])

        # Output convolution to refine final output
        self.output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def build(self, input_shape):
        # Build sub-layers
        self.avg_pool.build(input_shape)
        avg_pool_output_shape = self.avg_pool.compute_output_shape(input_shape)
        self.transformation_model.build(avg_pool_output_shape)
        self.output_layer.build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.output_layer.compute_output_shape(input_shape)

    def call(self, inputs):
        # Downscale input
        downscaled_inputs = self.avg_pool(inputs)

        # Predict transformation parameters
        transformations = self.transformation_model(downscaled_inputs)

        # Apply spatial transformation
        grids = generate_normalized_homo_meshgrids(inputs)
        reprojected_grids = transform_grids(transformations, grids, inputs)

        # Perform bilinear sampling
        sampled_output = bilinear_sample(inputs, reprojected_grids)

        # Refine output using convolution
        output = self.output_layer(sampled_output)

        return output

    def get_config(self):
        base_config = super(SpatialTransformerInputHead, self).get_config()
        config = {'downscaling': self.downscaling}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Helper function to generate normalized homogeneous grid
@tf.function
@keras.saving.register_keras_serializable(package="qr_model")
def generate_normalized_homo_meshgrids(inputs):
    batch_size = tf.shape(inputs)[0]
    _, H, W, _ = inputs.shape
    x_range = tf.range(W, dtype=tf.float32)  # Ensure float32
    y_range = tf.range(H, dtype=tf.float32)  # Ensure float32
    x_mesh, y_mesh = tf.meshgrid(x_range, y_range)

    # Normalize to range [-1, 1]
    x_mesh = (x_mesh / W - 0.5) * 2
    y_mesh = (y_mesh / H - 0.5) * 2

    # Create homogeneous coordinates
    ones_mesh = tf.ones_like(x_mesh, dtype=tf.float32)

    # Stack to form homogeneous coordinates
    homogeneous_grid = tf.stack([x_mesh, y_mesh, ones_mesh], axis=-1)  # Shape: [H, W, 3]

    # Reshape to [1, H * W, 3]
    homogeneous_grid = tf.reshape(homogeneous_grid, (1, -1, 3))

    # Tile to match batch size
    return tf.tile(homogeneous_grid, [batch_size, 1, 1])  # Shape: [batch_size, H * W, 3]




# Helper function to apply affine transformations to grids
@tf.function
@keras.saving.register_keras_serializable(package="qr_model")
def transform_grids(transformations, grids, inputs):
    batch_size = tf.shape(inputs)[0]

    # Reshape to apply affine transformation
    transformations = tf.reshape(transformations, [batch_size, 2, 3])
    grids = tf.cast(grids, tf.float32)

    # Apply affine transformation
    reprojected_grids = tf.matmul(transformations, grids, transpose_b=True)

    # Re-normalize grids to the input dimensions
    reprojected_grids = (reprojected_grids + 1) / 2  # Scale from [-1, 1] to [0, 1]

    return reprojected_grids


@tf.function
@keras.saving.register_keras_serializable(package="qr_model")
def generate_four_neighbors_from_reprojection(inputs, reprojected_grids):
    _, H, W, _ = inputs.shape
    x, y = tf.split(reprojected_grids, 2, axis=-1)

    x1 = tf.floor(x)
    x1 = tf.cast(x1, tf.int32)
    x2 = x1 + tf.constant(1)

    y1 = tf.floor(y)
    y1 = tf.cast(y1, tf.int32)
    y2 = y1 + tf.constant(1)

    y_max = tf.constant(H - 1, dtype=tf.int32)
    x_max = tf.constant(W - 1, dtype=tf.int32)
    zero = tf.zeros([1], dtype=tf.int32)

    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)
    x2_safe = tf.clip_by_value(x2, zero, x_max)
    y2_safe = tf.clip_by_value(y2, zero, y_max)

    return x1_safe, y1_safe, x2_safe, y2_safe


# Bilinear sampling for the spatial transformer
@tf.function
@keras.saving.register_keras_serializable(package="qr_model")
def bilinear_sample(inputs, reprojected_grids):
    x1, y1, x2, y2 = generate_four_neighbors_from_reprojection(inputs, reprojected_grids)

    # Get batch size and input dimensions
    batch_size = tf.shape(inputs)[0]
    H, W, channels = inputs.shape[1:4]  # Shape: [H, W, channels]

    # Create indices for pixel sampling
    x1y1 = tf.concat([tf.expand_dims(y1, -1), tf.expand_dims(x1, -1)], axis=-1)  # Shape: [batch_size, num_samples, 2]
    x1y2 = tf.concat([tf.expand_dims(y2, -1), tf.expand_dims(x1, -1)], axis=-1)
    x2y1 = tf.concat([tf.expand_dims(y1, -1), tf.expand_dims(x2, -1)], axis=-1)
    x2y2 = tf.concat([tf.expand_dims(y2, -1), tf.expand_dims(x2, -1)], axis=-1)

    # Gather pixel values from inputs
    pixel_x1y1 = tf.gather_nd(inputs, x1y1, batch_dims=0)
    pixel_x1y2 = tf.gather_nd(inputs, x1y2, batch_dims=0)
    pixel_x2y1 = tf.gather_nd(inputs, x2y1, batch_dims=0)
    pixel_x2y2 = tf.gather_nd(inputs, x2y2, batch_dims=0)

    # Continue with bilinear sampling calculation
    x, y = tf.split(reprojected_grids, 2, axis=-1)
    wx = tf.concat([tf.cast(x2, tf.float32) - x, x - tf.cast(x1, tf.float32)], -1)
    wx = tf.expand_dims(wx, -2)
    wy = tf.concat([tf.cast(y2, tf.float32) - y, y - tf.cast(y1, tf.float32)], -1)
    wy = tf.expand_dims(wy, -1)

    # Concatenate pixel values into a single tensor
    Q = tf.concat([pixel_x1y1, pixel_x1y2, pixel_x2y1, pixel_x2y2], -1)

    # Check the shape of Q before reshaping
    Q_shape = tf.shape(Q)
    print("Shape of Q:", Q_shape)  # Debugging statement

    # Reshape Q properly; verify dimensions
    if Q_shape[2] == 4:  # Assuming the last dimension is 4 for bilinear sampling
        Q = tf.reshape(Q, (Q_shape[0], Q_shape[1], 2, 2))  # Adjust if dimensions allow
    else:
        raise ValueError("Unexpected shape for Q, cannot reshape.")

    r = wx @ Q @ wy
    r = tf.reshape(r, (batch_size, H, W, 1))  # Use batch_size directly
    return r




# Function to gather pixel values for bilinear sampling
@tf.function
def gather_pixel_value(inputs, x, y):
    indices = tf.stack([y, x], axis=-1)
    return tf.gather_nd(inputs, indices, batch_dims=1)
