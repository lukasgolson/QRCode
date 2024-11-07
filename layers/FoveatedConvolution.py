import tensorflow as tf
from tensorflow.keras import layers


class FoveatedConvolutionLayer(layers.Layer):
    def __init__(self, fovea_size=(128, 128), reduction_factor=4, **kwargs):
        super(FoveatedConvolutionLayer, self).__init__(**kwargs)
        self.fovea_size = fovea_size
        self.reduction_factor = reduction_factor

        self.dilated_conv = layers.Conv2D(8, (3, 3), strides=1, dilation_rate=2, padding='valid', activation='relu')

        # Define stride convolution without dilation
        self.stride_conv = layers.Conv2D(8, (3, 3), strides=2, dilation_rate=1, padding='valid', activation='relu')

        # Center and transposed convolutions as before
        self.conv_center = layers.Conv2D(8, (3, 3), strides=1, dilation_rate=1, padding='valid', activation='relu')
        self.transconv_half = layers.Conv2DTranspose(8, (3, 3), strides=2, padding='valid', activation='relu')
        self.transconv_quarter = layers.Conv2DTranspose(8, (3, 3), strides=4, padding='valid', activation='relu')

    @tf.function
    def call(self, inputs):
        # Get input dimensions
        input_shape = tf.shape(inputs)
        h, w = input_shape[1], input_shape[2]

        # Calculate starting points for the central crop
        start_h = (h - self.fovea_size[0]) // 2
        start_w = (w - self.fovea_size[1]) // 2

        # Central crop (fovea area)
        crop_center = inputs[:, start_h:start_h + self.fovea_size[0], start_w:start_w + self.fovea_size[1], :]

        # Half crop (central region of half the fovea size)
        half_h, half_w = self.fovea_size[0] // 2, self.fovea_size[1] // 2
        start_h_half = (h - half_h) // 2
        start_w_half = (w - half_w) // 2
        crop_half = inputs[:, start_h_half:start_h_half + half_h, start_w_half:start_w_half + half_w, :]

        # Quarter crop (central region of quarter the fovea size)
        quarter_h, quarter_w = self.fovea_size[0] // 4, self.fovea_size[1] // 4
        start_h_quarter = (h - quarter_h) // 2
        start_w_quarter = (w - quarter_w) // 2
        crop_quarter = inputs[:, start_h_quarter:start_h_quarter + quarter_h,
                       start_w_quarter:start_w_quarter + quarter_w, :]

        # Apply dilated convolution followed by a stride-based downsample for the peripheral region
        x1_dilated = self.dilated_conv(inputs)
        x1 = self.stride_conv(x1_dilated)

        # Apply the central convolution to a central crop without dilation or stride
        x2 = self.conv_center(crop_center)

        # Apply transpose convolutions on smaller central crops
        x3 = self.transconv_half(crop_half)

        x4 = self.transconv_quarter(crop_quarter)

        # Resize all outputs back to input size for concatenation
        inputs_size = tf.shape(inputs)[1:3]
        x1_resized = tf.image.resize(x1, inputs_size, method='nearest')
        x2_resized = tf.image.resize(x2, inputs_size, method='nearest')
        x3_resized = tf.image.resize(x3, inputs_size, method='nearest')
        x4_resized = tf.image.resize(x4, inputs_size, method='nearest')

        # Concatenate along channels as the final output
        combined_output = tf.concat([x1_resized, x2_resized, x3_resized, x4_resized], axis=-1)
        return combined_output

    def get_config(self):
        config = super(FoveatedConvolutionLayer, self).get_config()
        config.update({
            'fovea_size': self.fovea_size,
            'reduction_factor': self.reduction_factor
        })
        return config
