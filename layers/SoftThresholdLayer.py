import tensorflow as tf
from tensorflow.keras.layers import Layer


class SoftThresholdLayer(Layer):
    def __init__(self, initial_threshold=0.5, **kwargs):
        super(SoftThresholdLayer, self).__init__(**kwargs)
        self.initial_threshold = initial_threshold

    def build(self, input_shape):
        # Create a threshold parameter for each channel
        self.threshold = self.add_weight(
            name='threshold',
            shape=(input_shape[-1],),  # One threshold per channel
            initializer=tf.keras.initializers.Constant(self.initial_threshold),
            trainable=True
        )
        super(SoftThresholdLayer, self).build(input_shape)

    def call(self, inputs):
        # Apply soft thresholding independently to each channel
        # inputs is of shape (batch_size, height, width, channels)
        soft_thresholded = tf.maximum(0.0, inputs - tf.expand_dims(self.threshold, axis=0)) - \
                           tf.maximum(0.0, -inputs - tf.expand_dims(self.threshold, axis=0))
        return soft_thresholded

    def get_config(self):
        config = super(SoftThresholdLayer, self).get_config()
        config.update({'initial_threshold': self.initial_threshold})
        return config
