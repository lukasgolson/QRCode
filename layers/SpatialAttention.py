import keras
from keras.src import layers
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply


@keras.saving.register_keras_serializable(package="qr_model", name="SpatialAttention")
class SpatialAttention(layers.Layer):
    """
    A Keras layer that applies spatial attention to the input features.
    It computes attention weights based on the global average pooling of the input
    and then multiplies the input with these weights to enhance important features.

    Attributes:
        dense (Dense): A Dense layer for generating attention weights.
    """

    def __init__(self, **kwargs):
        """
        Initializes the SpatialAttention layer.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.dense = None  # To be defined in build method

    def build(self, input_shape):
        """
        Builds the layer by initializing the Dense layer for attention weights.

        Args:
            input_shape (tuple): Shape of the input tensor.
                Expected shape is (batch_size, height, width, channels).
        """
        self.dense = Dense(input_shape[-1], activation='sigmoid')

    def call(self, inputs):
        """
        Applies the spatial attention mechanism to the inputs.

        Args:
            inputs (tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tensor: The output tensor with spatial attention applied.
        """
        # Compute global average pooling across the spatial dimensions
        avg_pooled = GlobalAveragePooling2D()(inputs)

        # Reshape to add spatial dimensions for multiplication
        reshaped = Reshape((1, 1, inputs.shape[-1]))(avg_pooled)

        # Generate attention weights
        attention_weights = self.dense(reshaped)

        # Multiply the input with the attention weights
        output = Multiply()([inputs, attention_weights])

        return output

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary containing layer parameters.
        """
        base_config = super(SpatialAttention, self).get_config()
        config = {}
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
