import keras
from keras import Layer
from keras.src import initializers
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply


@keras.saving.register_keras_serializable(package="qr_model", name="SqueezeExcitation")
class SqueezeExcitation(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.dense2 = None
        self.dense1 = None
        self.channels = None
        self.ratio = ratio

    def build(self, input_shape):
        # Define the number of channels
        self.channels = input_shape[-1]
        # Create two Dense layers for excitation
        self.dense1 = Dense(self.channels // self.ratio, activation='relu')
        self.dense2 = Dense(self.channels, activation='mish', kernel_initializer="he_normal")

    def call(self, inputs):
        # Squeeze: Global Average Pooling
        se = GlobalAveragePooling2D()(inputs)
        # Excitation: Dense layers
        se = self.dense1(se)
        se = self.dense2(se)
        # Reshape to match input shape
        se = Reshape((1, 1, self.channels))(se)
        # Scale the input
        return Multiply()([inputs, se])

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({"ratio": self.ratio})
        return config
