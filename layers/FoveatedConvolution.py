import tensorflow as tf
from tensorflow.keras import layers


class FoveatedConvolutionLayer(layers.Layer):
    def __init__(self, fovea_size=(64, 64), reduction_factor=4, **kwargs):
        super(FoveatedConvolutionLayer, self).__init__(**kwargs)
        self.fovea_size = fovea_size  # Central region size (fovea)
        self.reduction_factor = reduction_factor  # Factor for peripheral resolution decrease

    def build(self, input_shape):
        # Define the kernels for each convolutional layer
        self.conv_dilation = layers.Conv2D(8, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu')
        self.conv_stride = layers.Conv2D(8, (3, 3), strides=2, dilation_rate=1, padding='same', activation='relu')

        self.conv2 = layers.Conv2D(8, (3, 3), strides=1, dilation_rate=1, padding='same', activation='relu')
        self.transconv1 = layers.Conv2DTranspose(8, (3, 3), strides=2, dilation_rate=1, padding='same',
                                                 activation='relu')
        self.transconv2 = layers.Conv2DTranspose(8, (3, 3), strides=4, dilation_rate=1, padding='same',
                                                 activation='relu')

        super(FoveatedConvolutionLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        x1_dilated = self.conv_dilation(inputs)

        # Apply stride second (stride 2, dilation 1)
        x1 = self.conv_stride(x1_dilated)

        # Apply the second convolution (stride 1, dilation 1) to a central crop
        crop2 = inputs[:, :self.fovea_size[0], :self.fovea_size[1], :]
        x2 = self.conv2(crop2)

        # Apply transpose convolution on a smaller central crop
        crop3 = inputs[:, :self.fovea_size[0] // 2, :self.fovea_size[1] // 2, :]
        x3 = self.transconv1(crop3)

        # Apply a final transpose convolution on an even smaller central crop
        crop4 = inputs[:, :self.fovea_size[0] // 4, :self.fovea_size[1] // 4, :]
        x4 = self.transconv2(crop4)

        inputs_size = tf.shape(inputs)[1:3]  # Get height and width of x1
        x1_resized = tf.image.resize(x1, inputs_size)
        x2_resized = tf.image.resize(x2, inputs_size)
        x3_resized = tf.image.resize(x3, inputs_size)
        x4_resized = tf.image.resize(x4, inputs_size)

        # Combine the results from each convolutional layer
        combined = tf.concat([x1_resized, x2_resized, x3_resized, x4_resized], axis=-1)


        return combined

    def get_config(self):
        base_config = super(FoveatedConvolutionLayer, self).get_config()
        config = {
            'fovea_size': self.fovea_size,
            'reduction_factor': self.reduction_factor
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
