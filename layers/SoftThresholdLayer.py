import tensorflow as tf
from tensorflow.keras.layers import Layer


class SoftThresholdLayer(Layer):
    def __init__(self, initial_threshold=0.5, min_threshold=0.01, max_threshold=1.0, **kwargs):
        super(SoftThresholdLayer, self).__init__(**kwargs)
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def build(self, input_shape):
        self.threshold = self.add_weight(
            name='threshold',
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.initial_threshold),
            constraint=tf.keras.constraints.MinMaxNorm(self.min_threshold, self.max_threshold),
            trainable=True
        )
        super(SoftThresholdLayer, self).build(input_shape)

    def call(self, inputs):
        def smooth_max(x):
            return tf.log(1 + tf.exp(x))

            # Calculate the smooth thresholded values

        positive_part = smooth_max(inputs - tf.expand_dims(self.threshold, axis=0))
        negative_part = smooth_max(-inputs - tf.expand_dims(self.threshold, axis=0))

        soft_thresholded = positive_part - negative_part

        return soft_thresholded

    def get_config(self):
        config = super(SoftThresholdLayer, self).get_config()
        config.update({
            'initial_threshold': self.initial_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
        })
        return config
