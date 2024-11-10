import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

class HarmonicConv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, harmonic_ratio=1.0, **kwargs):
        super(HarmonicConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.harmonic_ratio = harmonic_ratio
        self.conv_layer = layers.Conv2D(filters, kernel_size, padding="same")
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
        # Batched FFT on the entire input tensor in complex form
        fft_input = tf.signal.fft2d(tf.cast(inputs, tf.complex64))

        # Split into real and imaginary parts
        real_fft = tf.math.real(fft_input)
        imag_fft = tf.math.imag(fft_input)

        # Apply harmonic filter separately on real and imaginary components
        real_filtered = tf.nn.conv2d(real_fft, self.harmonic_filters, strides=[1, self.stride, self.stride, 1], padding="SAME")
        imag_filtered = tf.nn.conv2d(imag_fft, self.harmonic_filters, strides=[1, self.stride, self.stride, 1], padding="SAME")

        # Reconstruct the filtered complex tensor
        filtered_input = tf.complex(real_filtered, imag_filtered)

        # Inverse FFT to return to spatial domain, taking only the real part
        filtered_input = tf.signal.ifft2d(filtered_input)
        filtered_input = tf.math.real(filtered_input)  # Only take the real part

        # Final convolution without activation
        output = self.conv_layer(filtered_input)

        return output

    def compute_output_shape(self, input_shape):
        # Ensure the output shape matches the input spatial dimensions and specified filters
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super(HarmonicConv2D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "harmonic_ratio": self.harmonic_ratio,
        })
        return config
