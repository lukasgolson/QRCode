import keras
import numpy as np
from keras import Layer
from keras.src.initializers import Constant
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf


@keras.saving.register_keras_serializable(package="qr_model", name="SpatialTransformerInputHead")
class SpatialTransformerInputHead(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialTransformerInputHead, self).__init__(**kwargs)

        # Localization head layers
        self.conv1 = Conv2D(14, (5, 5), padding='valid', activation='relu')
        self.pool1 = MaxPooling2D((2, 2), strides=2)
        self.conv2 = Conv2D(32, (5, 5), padding='valid', activation='relu')
        self.pool2 = MaxPooling2D((2, 2), strides=2)
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(84, activation='relu')

        bias_initializer = Constant([1, 0, 0, 0, 1, 0])
        self.internal_output_layer = Dense(6, activation='linear', kernel_initializer='zeros', bias_initializer=bias_initializer)

        self.output_layer = Conv2D(1, (1, 1), activation='sigmoid')  # Adjust as necessary for your output

    def build(self, input_shape):
        # Relying on Keras' automatic shape inference
        super(SpatialTransformerInputHead, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.output_layer.compute_output_shape(input_shape)

    def call(self, inputs):
        # Compute transformations using localization head
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        transformations = self.internal_output_layer(x)

        # Apply spatial transformation
        grids = self.generate_normalized_homo_meshgrids(inputs)
        reprojected_grids = self.transform_grids(transformations, grids, inputs)

        sampled_output = self.bilinear_sample(inputs, reprojected_grids)

        output = self.output_layer(sampled_output)  # Use final output layer to ensure shape

        return output

    def get_config(self):
        base_config = super(SpatialTransformerInputHead, self).get_config()
        config = {
            "conv1": keras.layers.serialize(self.conv1),
            "pool1": keras.layers.serialize(self.pool1),
            "conv2": keras.layers.serialize(self.conv2),
            "pool2": keras.layers.serialize(self.pool2),
            "flatten": keras.layers.serialize(self.flatten),
            "dense1": keras.layers.serialize(self.dense1),
            "dropout": keras.layers.serialize(self.dropout),
            "dense2": keras.layers.serialize(self.dense2),
            "internal_output_layer": keras.layers.serialize(self.internal_output_layer),
            "output_layer": keras.layers.serialize(self.output_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config['conv1'] = keras.layers.deserialize(config['conv1'])
        config['pool1'] = keras.layers.deserialize(config['pool1'])
        config['conv2'] = keras.layers.deserialize(config['conv2'])
        config['pool2'] = keras.layers.deserialize(config['pool2'])
        config['flatten'] = keras.layers.deserialize(config['flatten'])
        config['dense1'] = keras.layers.deserialize(config['dense1'])
        config['dropout'] = keras.layers.deserialize(config['dropout'])
        config['dense2'] = keras.layers.deserialize(config['dense2'])
        config['internal_output_layer'] = keras.layers.deserialize(config['internal_output_layer'])
        config['output_layer'] = keras.layers.deserialize(config['output_layer'])

        return cls(**config)

    def generate_normalized_homo_meshgrids(self, inputs):
        batch_size = tf.shape(inputs)[0]
        _, H, W, _ = inputs.shape
        x_range = tf.range(W)
        y_range = tf.range(H)
        x_mesh, y_mesh = tf.meshgrid(x_range, y_range)
        x_mesh = (x_mesh / W - 0.5) * 2
        y_mesh = (y_mesh / H - 0.5) * 2
        y_mesh = tf.reshape(y_mesh, (*y_mesh.shape, 1))
        x_mesh = tf.reshape(x_mesh, (*x_mesh.shape, 1))
        ones_mesh = tf.ones_like(x_mesh)
        homogeneous_grid = tf.concat([x_mesh, y_mesh, ones_mesh], -1)
        homogeneous_grid = tf.reshape(homogeneous_grid, (-1, 3, 1))
        homogeneous_grid = tf.cast(homogeneous_grid, tf.float32)
        homogeneous_grid = tf.expand_dims(homogeneous_grid, 0)
        return tf.tile(homogeneous_grid, [batch_size, 1, 1, 1])

    def transform_grids(self, transformations, grids, inputs):
        trans_matrices = tf.reshape(transformations, (-1, 2, 3))
        batch_size = tf.shape(trans_matrices)[0]
        gs = tf.squeeze(grids, -1)
        reprojected_grids = tf.matmul(trans_matrices, gs, transpose_b=True)
        reprojected_grids = (tf.linalg.matrix_transpose(reprojected_grids) + 1) * 0.5
        _, H, W, _ = inputs.shape
        return tf.multiply(reprojected_grids, [W, H])

    def generate_four_neighbors_from_reprojection(self, inputs, reprojected_grids):
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

    def bilinear_sample(self, inputs, reprojected_grids):
        x1, y1, x2, y2 = self.generate_four_neighbors_from_reprojection(inputs, reprojected_grids)
        x1y1 = tf.concat([y1, x1], -1)
        x1y2 = tf.concat([y2, x1], -1)
        x2y1 = tf.concat([y1, x2], -1)
        x2y2 = tf.concat([y2, x2], -1)
        pixel_x1y1 = tf.gather_nd(inputs, x1y1, batch_dims=1)
        pixel_x1y2 = tf.gather_nd(inputs, x1y2, batch_dims=1)
        pixel_x2y1 = tf.gather_nd(inputs, x2y1, batch_dims=1)
        pixel_x2y2 = tf.gather_nd(inputs, x2y2, batch_dims=1)
        x, y = tf.split(reprojected_grids, 2, axis=-1)
        wx = tf.concat([tf.cast(x2, tf.float32) - x, x - tf.cast(x1, tf.float32)], -1)
        wx = tf.expand_dims(wx, -2)
        wy = tf.concat([tf.cast(y2, tf.float32) - y, y - tf.cast(y1, tf.float32)], -1)
        wy = tf.expand_dims(wy, -1)
        Q = tf.concat([pixel_x1y1, pixel_x1y2, pixel_x2y1, pixel_x2y2], -1)
        Q_shape = tf.shape(Q)
        Q = tf.reshape(Q, (Q_shape[0], Q_shape[1], 2, 2))
        r = wx @ Q @ wy
        _, H, W, channels = inputs.shape
        r = tf.reshape(r, (-1, H, W, 1))
        return r
