import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

class HarmonicConv2D(Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(HarmonicConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_layer = layers.Conv2D(filters, kernel_size, padding="same")
        self.activation_layer = layers.Activation(activation)
        self.harmonic_mask = None

    def build(self, input_shape):
        self.harmonic_mask = self.add_weight(
            shape=(input_shape[1], input_shape[2], input_shape[3]),
            initializer="uniform",
            trainable=True,
            name="harmonic_mask"
        )
        super(HarmonicConv2D, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Apply FFT
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))

        # Element-wise multiplication with harmonic_mask
        harmonic_fft = fft * tf.cast(self.harmonic_mask, tf.complex64)

        # Apply inverse FFT and take the real part
        filtered_input = tf.signal.ifft2d(harmonic_fft)
        filtered_input = tf.math.real(filtered_input)  # Only take the real part

        # Convolution and activation
        convoluted = self.conv_layer(filtered_input)
        output = self.activation_layer(convoluted)

        return output

    def get_config(self):
        config = super(HarmonicConv2D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config
