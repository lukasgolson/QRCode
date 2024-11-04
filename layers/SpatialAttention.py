import keras
from keras.layers import Conv2D, Multiply, Activation, Add, BatchNormalization, Concatenate
import tensorflow as tf

@keras.saving.register_keras_serializable(package="qr_model", name="SpatialAttention")
class SpatialAttention(keras.layers.Layer):
    def __init__(self, num_layers=3, initial_filters=3, filter_step=3, use_residual=True, **kwargs):
        """
        Initializes the SpatialAttention layer.

        Args:
            num_layers (int): Number of convolution layers.
            initial_filters (int): Number of filters for the first convolution layer.
            filter_step (int): Increment of filters for each subsequent layer.
            use_residual (bool): If True, enables the skip connection.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.initial_filters = initial_filters
        self.filter_step = filter_step
        self.use_residual = use_residual

        self.convs = []
        self.activations = []

        for i in range(num_layers):
            filters = initial_filters + i * filter_step
            self.convs.append(Conv2D(filters=filters, kernel_size=3, padding='same', name=f'conv{i+1}'))
            self.activations.append(Activation('relu'))

        self.conv_pooling = Conv2D(filters=1, kernel_size=1, padding='same', name='conv_pooling')
        self.final_activation = Activation('sigmoid')
        self.concatenate = Concatenate(axis=-1)
        self.batch_norm = BatchNormalization(name='batch_norm')
        self.residual_activation = Activation('relu')

    def build(self, input_shape):
        attention_map_shapes = [input_shape]
        for conv in self.convs:
            conv.build(attention_map_shapes[-1])
            attention_map_shapes.append(conv.compute_output_shape(attention_map_shapes[-1]))

        self.concatenate.build(attention_map_shapes[1:])
        concatenated_shape = self.concatenate.compute_output_shape(attention_map_shapes[1:])
        self.conv_pooling.build(concatenated_shape)

        if self.use_residual:
            self.batch_norm.build(input_shape)

        super(SpatialAttention, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        attention_maps = [inputs]

        for conv, activation in zip(self.convs, self.activations):
            x = conv(attention_maps[-1])
            x = activation(x)
            attention_maps.append(x)

        x = self.concatenate(attention_maps[1:])
        x = self.conv_pooling(x)
        x = self.final_activation(x)
        x = Multiply()([x, inputs])

        if self.use_residual:
            x = Add()([inputs, x])
            x = self.batch_norm(x)
            x = self.residual_activation(x)

        return x

    def get_config(self):
        base_config = super(SpatialAttention, self).get_config()
        config = {
            'num_layers': self.num_layers,
            'initial_filters': self.initial_filters,
            'filter_step': self.filter_step,
            'use_residual': self.use_residual
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
