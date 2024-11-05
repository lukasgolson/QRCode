import keras
from keras import Layer
from keras.src import initializers, layers
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply, Add, BatchNormalization, Activation
import tensorflow as tf


@keras.saving.register_keras_serializable(package="qr_model", name="SqueezeExcitation")
class SqueezeExcitation(Layer):
    def __init__(self, ratio=16, use_residual=True, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)

        self.activation = None
        self.dense2 = None
        self.dense1 = None
        self.channels = None
        self.ratio = ratio
        self.use_residual = use_residual

        self.multiply_layer = Multiply()



        self.add_layer = Add()

        if self.use_residual:
            self.batch_norm = BatchNormalization(name='batch_norm')
            self.residual_activation = layers.LeakyReLU()

    def build(self, input_shape):
        # Define the number of channels
        self.channels = input_shape[-1]
        # Create two Dense layers for excitation
        self.dense1 = Dense(self.channels // self.ratio)
        self.activation = layers.LeakyReLU()

        self.dense2 = Dense(self.channels, activation='sigmoid')

        if self.use_residual:
            self.batch_norm.build(input_shape)
            batch_norm_shape = self.batch_norm.compute_output_shape(input_shape)
            self.residual_activation.build(batch_norm_shape)

    @tf.function
    def call(self, inputs):

        # Squeeze: Global Average Pooling
        se = GlobalAveragePooling2D()(inputs)
        # Excitation: Dense layers
        se = self.dense1(se)
        se = self.activation(se)
        se = self.dense2(se)
        # Reshape to match input shape
        se = Reshape((1, 1, self.channels))(se)

        se = self.multiply_layer([inputs, se])

        if self.use_residual:
            se = self.add_layer([inputs, se])
            se = self.batch_norm(se)
            se = self.residual_activation(se)

        # Scale the input
        return se

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({"ratio": self.ratio, "use_residual": self.use_residual})
        return config
