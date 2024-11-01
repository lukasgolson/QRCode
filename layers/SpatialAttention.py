import keras
from keras.layers import Conv2D, Multiply, Activation, Add, BatchNormalization, Concatenate


@keras.saving.register_keras_serializable(package="qr_model", name="SpatialAttention")
class SpatialAttention(keras.layers.Layer):
    """
    A Keras layer that applies spatial attention to the input features using CNN.
    It computes attention weights based on convolutional operations on the input
    and then multiplies the input with these weights to enhance important features.
    An optional skip connection can also be included.
    """

    def __init__(self, use_residual=True, **kwargs):
        """
        Initializes the SpatialAttention layer.

        Args:
            use_residual (bool): If True, enables the skip connection.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        # Initialize layers in __init__
        self.conv1 = Conv2D(filters=1, kernel_size=2, padding='same', activation='mish', name='conv1')
        self.conv2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='mish', name='conv2')
        self.conv3 = Conv2D(filters=1, kernel_size=5, padding='same', activation='mish', name='conv3')

        self.conv_pooling = None

        self.concatenate = Concatenate(axis=-1)
        self.batch_norm = BatchNormalization(name='batch_norm')
        self.activation = Activation('mish', name='activation')  # 'mish' can be defined if not available in Keras
        self.use_residual = use_residual

    def build(self, input_shape):
        """
        Builds the layer by initializing convolutional layers for generating attention weights.

        Args:
            input_shape (tuple): Shape of the input tensor (batch_size, height, width, channels).
        """
        self.conv_pooling = Conv2D(filters=input_shape[-1], kernel_size=1, padding='same', activation='sigmoid',
                                   name='conv_pooling')

        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        self.conv3.build(input_shape)

        attention_map1_shape = self.conv1.compute_output_shape(input_shape)
        attention_map2_shape = self.conv2.compute_output_shape(input_shape)
        attention_map3_shape = self.conv3.compute_output_shape(input_shape)

        self.concatenate.build([attention_map1_shape, attention_map2_shape, attention_map3_shape])

        concatenate_shape = self.concatenate.compute_output_shape(
            [attention_map1_shape, attention_map2_shape, attention_map3_shape])

        self.conv_pooling.build(concatenate_shape)

        conv_pooling_shape = self.conv_pooling.compute_output_shape(concatenate_shape)

        if self.use_residual:
            self.batch_norm.build(conv_pooling_shape)

            batch_norm_shape = self.batch_norm.compute_output_shape(conv_pooling_shape)

            self.activation.build(batch_norm_shape)

        super(SpatialAttention, self).build(input_shape)

        # Call the parent class's build method

    def call(self, inputs):
        """
        Applies the spatial attention mechanism to the inputs.

        Args:
            inputs (tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tensor: The output tensor with spatial attention applied, with optional skip connection.
        """
        # Compute attention maps using convolutional operations
        attention_map1 = self.conv1(inputs)
        attention_map2 = self.conv2(inputs)
        attention_map3 = self.conv3(inputs)

        # Concatenate the attention maps along the channel axis
        x = self.concatenate([attention_map1, attention_map2, attention_map3])

        # Apply 1x1 convolution to reduce the concatenated maps to the same depth as inputs
        x = self.conv_pooling(x)

        # Multiply the input with the combined attention weights
        x = Multiply()([inputs, x])

        # If skip connection is enabled, add the input to the output
        if self.use_residual:
            x = Add()([inputs, x])  # Skip connection

            # Apply batch normalization and activation
            x = self.batch_norm(x)
            x = self.activation(x)

        return x

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary containing layer parameters.
        """
        base_config = super(SpatialAttention, self).get_config()
        config = {'use_residual': self.use_residual}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of the layer from its configuration.

        Args:
            config (dict): Configuration dictionary to recreate the layer.

        Returns:
            SpatialAttention: An instance of the SpatialAttention layer.
        """
        return cls(**config)
