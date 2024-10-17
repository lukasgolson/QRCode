import keras
from keras import layers
from keras.layers import Conv2D, Multiply, Activation, Add, BatchNormalization
from keras.src.layers import Dense


@keras.saving.register_keras_serializable(package="qr_model", name="SpatialAttention")
class SpatialAttention(layers.Layer):
    """
    A Keras layer that applies spatial attention to the input features using CNN.
    It computes attention weights based on convolutional operations on the input
    and then multiplies the input with these weights to enhance important features.
    An optional skip connection can also be included.
    """

    def __init__(self, use_skip_connection=False, **kwargs):
        """
        Initializes the CNNSpatialAttention layer.

        Args:
            use_skip_connection (bool): If True, enables the skip connection.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.use_skip_connection = use_skip_connection

    def build(self, input_shape):
        """
        Builds the layer by initializing convolutional layers for generating attention weights.

        Args:
            input_shape (tuple): Shape of the input tensor (batch_size, height, width, channels).
        """
        # Initialize convolutional layers for attention weights
        self.conv1 = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', name='conv1')
        self.conv2 = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', name='conv2')
        self.conv_pooling = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid', name='conv_pooling')

        # Initialize dense layer for final output
        self.dense = Dense(input_shape[-1], activation='sigmoid', name='dense')

        # Initialize batch normalization
        self.batch_norm = BatchNormalization(name='batch_norm')

        # Mish activation layer
        self.activation = layers.Activation('mish', name='activation')

        # Call build on all sublayers
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        self.conv_pooling.build(input_shape)
        self.dense.build((None, input_shape[-1]))  # dense layer input shape is (None, channels)
        self.batch_norm.build(input_shape)

        # Set trainable weights for the layers
        self.built = True

    def call(self, inputs):
        """
        Applies the spatial attention mechanism to the inputs.

        Args:
            inputs (tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tensor: The output tensor with spatial attention applied, with optional skip connection.
        """
        # Compute attention maps using two convolutional operations
        attention_map1 = self.conv1(inputs)
        attention_map2 = self.conv2(inputs)

        # Combine the attention maps
        combined_attention = Activation('sigmoid')(attention_map1 + attention_map2)

        # Apply pooling to the combined attention map
        combined_attention = self.conv_pooling(combined_attention)

        combined_attention = self.dense(combined_attention)

        # Multiply the input with the combined attention weights
        output = Multiply()([inputs, combined_attention])

        # If skip connection is enabled, add the input to the output
        if self.use_skip_connection:
            output = Add()([output, inputs])  # Skip connection

        output = self.batch_norm(output)
        output = self.activation(output)

        return output

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary containing layer parameters.
        """
        base_config = super(SpatialAttention, self).get_config()
        config = {
            'use_skip_connection': self.use_skip_connection,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of the layer from its configuration.

        Args:
            config (dict): Configuration dictionary to recreate the layer.

        Returns:
            CNNSpatialAttention: An instance of the CNNSpatialAttention layer.
        """
        return cls(**config)
