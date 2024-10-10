import keras
import tensorflow as tf

class Involution(keras.layers.Layer):
    """
    A custom Keras layer implementing the Involution operation.

    Involution is an efficient alternative to convolution that adapts its kernel to each input feature map,
    capturing local spatial relationships in a more flexible way.

    Args:
        channel (int): Number of input channels.
        group_number (int): Number of groups for group convolutions.
        kernel_size (int): Size of the kernel (KxK).
        stride (int): Stride for downsampling the input.
        reduction_ratio (int): Reduction ratio to reduce the number of channels when generating the kernel.
        name (str): Name of the layer.

    References:
        - Paper: https://arxiv.org/abs/2103.06255
    """
    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio, name):
        """
        Initialize the Involution layer with the specified parameters.

        Args:
            channel (int): Number of input channels.
            group_number (int): Number of groups for group convolutions.
            kernel_size (int): Size of the kernel (KxK).
            stride (int): Stride for downsampling the input.
            reduction_ratio (int): Reduction ratio for channel reduction during kernel generation.
            name (str): Name of the layer.
        """
        super().__init__(name=name)

        # Store the parameters for later use in the layer.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        """
        Build the internal layers for the Involution operation once the input shape is known.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        # Extract the height, width, and number of channels from the input shape.
        (_, height, width, num_channels) = input_shape

        # Downscale the height and width according to the stride.
        height = height // self.stride
        width = width // self.stride

        # Define a pooling layer to reduce the size of the input if the stride is greater than 1.
        # Otherwise, use the identity operation (no change to input).
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )

        # Kernel generation network: a sequence of two convolutional layers, batch normalization, and ReLU.
        # The first Conv2D reduces the channel size (controlled by the reduction_ratio).
        # The second Conv2D outputs a kernel of size K*K*G.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )

        # Reshape the generated kernel to shape (H, W, K*K, 1, G).
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height, width, self.kernel_size * self.kernel_size, 1, self.group_number
            )
        )

        # Reshape the extracted input patches to shape (H, W, K*K, C//G, G).
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height, width, self.kernel_size * self.kernel_size,
                num_channels // self.group_number, self.group_number
            )
        )

        # Reshape the output back to the original input shape (H, W, C).
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        """
        Perform the forward pass of the Involution layer, generating the adaptive kernel
        and applying it to the input patches.

        Args:
            x (tensor): Input tensor of shape (B, H, W, C), where B is batch size, H is height, W is width, and C is channels.

        Returns:
            tuple: A tuple of two elements:
                - output (tensor): Output tensor after applying the adaptive kernel, shape (B, H, W, C).
                - kernel (tensor): Generated kernel tensor, shape (B, H, W, K*K, 1, G).
        """
        # Apply stride layer (pooling or identity) to downsample the input.
        kernel_input = self.stride_layer(x)

        # Generate the kernel based on the input tensor.
        kernel = self.kernel_gen(kernel_input)

        # Reshape the kernel to match the format required for multiplication with input patches.
        kernel = self.kernel_reshape(kernel)

        # Extract sliding patches from the input tensor using the specified kernel size.
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches for multiplication with the kernel.
        input_patches = self.input_patches_reshape(input_patches)

        # Perform element-wise multiplication between the kernel and the input patches.
        output = tf.multiply(kernel, input_patches)

        # Sum over the kernel dimension (K*K) to reduce it.
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output to the original input shape (B, H, W, C).
        output = self.output_reshape(output)

        # Return the final output and the generated kernel (for potential visualization or analysis).
        return output, kernel
