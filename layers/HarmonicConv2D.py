import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

class HarmonicConv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, activation='relu', harmonic_ratio=1.0, **kwargs):
        super(HarmonicConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.harmonic_ratio = harmonic_ratio
        self.conv_layer = layers.Conv2D(filters, kernel_size, padding="same")
        self.activation_layer = layers.Activation(activation)
        self.harmonic_filters = None

    def build(self, input_shape):
        # Derive harmonic_kernel_size from kernel_size and harmonic_ratio
        harmonic_kernel_size = int(self.kernel_size * self.harmonic_ratio)
        harmonic_kernel_size = max(1, harmonic_kernel_size)  # Ensure it's at least 1

        # Initialize learnable harmonic filters
        self.harmonic_filters = self.add_weight(
            shape=(harmonic_kernel_size, harmonic_kernel_size, input_shape[-1], self.filters),
            initializer="uniform",
            trainable=True,
            name="harmonic_filters"
        )
        super(HarmonicConv2D, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Apply FFT on inputs
        fft_input = tf.signal.fft2d(tf.cast(inputs, tf.complex64))

        # Separate into real and imaginary components
        real_fft = tf.math.real(fft_input)
        imag_fft = tf.math.imag(fft_input)

        # Apply harmonic filter separately to real and imaginary components
        real_filtered = tf.nn.conv2d(real_fft, self.harmonic_filters, strides=[1, self.stride, self.stride, 1], padding="SAME")
        imag_filtered = tf.nn.conv2d(imag_fft, self.harmonic_filters, strides=[1, self.stride, self.stride, 1], padding="SAME")

        # Reconstruct the filtered complex output
        filtered_input = tf.complex(real_filtered, imag_filtered)

        # Apply inverse FFT to get back to spatial domain
        filtered_input = tf.signal.ifft2d(filtered_input)
        filtered_input = tf.math.real(filtered_input)  # Only take the real part for further processing

        # Convolution and activation
        convoluted = self.conv_layer(filtered_input)
        output = self.activation_layer(convoluted)

        return output

    def get_config(self):
        config = super(HarmonicConv2D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "harmonic_ratio": self.harmonic_ratio,
        })
        return config
