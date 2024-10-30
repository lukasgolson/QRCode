import keras
from keras import Layer
from keras.src import initializers
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply
from keras.src.utils.module_utils import tensorflow


@keras.saving.register_keras_serializable(package="qr_model", name="ExtractPatches")
class ExtractPatches(Layer):
    def __init__(self, ratio=2, **kwargs):
        super(ExtractPatches, self).__init__(**kwargs)

        self.ratio = ratio

        self.patch_size = (ratio, ratio)
        self.strides = self.patch_size

    def call(self, inputs):
        patches = tensorflow.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],  # [batch, height, width, channels]
            strides=[1, self.strides[0], self.strides[1], 1],  # [batch, height, width, channels]
            rates=[1, 1, 1, 1],  # [batch, height, width, channels]
            padding='VALID'
        )

        return patches


    def compute_output_shape(self, input_shape):
        # Input shape is expected to be (batch_size, height, width, channels)
        batch_size, input_height, input_width, channels = input_shape

        # Calculate the height and width of the output patches
        output_height = (input_height - self.patch_size[0]) // self.strides[0] + 1
        output_width = (input_width - self.patch_size[1]) // self.strides[1] + 1

        # The output shape will be (batch_size, output_height, output_width, (patch_size[0] * patch_size[1] * channels))
        output_shape = (batch_size, output_height, output_width, self.patch_size[0] * self.patch_size[1] * channels)
        return output_shape

    def get_config(self):
        config = super(ExtractPatches, self).get_config()
        config.update({"ratio": self.ratio})
        return config
