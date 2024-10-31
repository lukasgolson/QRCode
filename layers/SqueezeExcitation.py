import keras
from keras import Layer
from keras.src import initializers
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply, Add, BatchNormalization, Activation


@keras.saving.register_keras_serializable(package="qr_model", name="SqueezeExcitation")
class SqueezeExcitation(Layer):
    def __init__(self, ratio=16, use_residual=True, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.dense2 = None
        self.dense1 = None
        self.channels = None
        self.ratio = ratio
        self.use_residual = use_residual

        self.batch_norm = BatchNormalization(name='batch_norm')
        self.activation = Activation('mish', name='activation')  # 'mish' can be defined if not available in Keras

    def build(self, input_shape):
        # Define the number of channels
        self.channels = input_shape[-1]
        # Create two Dense layers for excitation
        self.dense1 = Dense(self.channels // self.ratio, activation='mish')
        self.dense2 = Dense(self.channels, activation='sigmoid')

        if self.use_residual:
            self.batch_norm.build(input_shape)
            batch_norm_shape = self.batch_norm.compute_output_shape(input_shape)
            self.activation.build(batch_norm_shape)

    def call(self, inputs):


        # Squeeze: Global Average Pooling
        se = GlobalAveragePooling2D()(inputs)
        # Excitation: Dense layers
        se = self.dense1(se)
        se = self.dense2(se)
        # Reshape to match input shape
        se = Reshape((1, 1, self.channels))(se)

        se = Multiply()([inputs, se])

        if self.use_residual:
            se = Add()([inputs, se])
            se = self.batch_norm(se)
            se = self.activation(se)

        # Scale the input
        return se

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({"ratio": self.ratio, "use_residual": self.use_residual})
        return config
