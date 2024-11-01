import keras
from keras import Model
from keras.src import layers

import tensorflow as tf
from keras import backend as K

import Dataset
from layers.SpatialTransformer import SpatialTransformer


def create_model(input_shape):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Apply spatial transformer to the input image
    x = SpatialTransformer()(inputs)

    # Encoder: A series of convolutional layers to process the noisy input image
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # Bottleneck layer for feature extraction
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # Decoder: Upsample and reconstruct the "cleaned" version of the image
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)

    # Final convolution to output a single-channel image, corrected version of input
    output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    # Define the model
    model = Model(inputs, output, name='qr_correction_model')

    return model


@tf.function
@keras.saving.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


@tf.function
@keras.saving.register_keras_serializable()
def combined_mse_ssim_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return alpha * mse + beta * ssim


def train_model(resolution=256, epochs=10):
    # Create the model
    input_shape = (resolution, resolution, 1)

    model = create_model(input_shape)

    dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution))

    # Compile the model
    model.compile(optimizer='adam', loss=combined_mse_ssim_loss, metrics=[ssim_loss])

    # Train the model
    model.fit(dataset, epochs=epochs, steps_per_epoch=250)

    # Save the model
    model.save('qr_correction_model.keras')


if __name__ == '__main__':
    train_model()
