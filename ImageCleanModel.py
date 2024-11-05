import argparse

import keras
from keras import Model
from keras.src import layers

import tensorflow as tf
from keras import backend as K
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adafactor
from tensorflow.keras.callbacks import TensorBoard

import Dataset
from layers.SoftThresholdLayer import SoftThresholdLayer
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformer
from layers.SqueezeExcitation import SqueezeExcitation


def Conv2DSkip(input_layer, filters, kernel_size, activation='relu', padding='same'):
    # Perform convolution on the input layer
    x = layers.Conv2D(filters, kernel_size, padding=padding)(input_layer)

    # Create a skip connection
    skip = input_layer

    # If the number of filters in the skip connection does not match, adjust it
    if input_layer.shape[-1] != filters:
        skip = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(input_layer)

    # Add the skip connection to the output of the convolutional layer
    x = layers.Add()([skip, x])

    # Apply the activation function after the addition
    return layers.Activation(activation)(x)


def create_model(input_shape):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Apply spatial transformer to the input image
    x = SpatialTransformer()(inputs)
    x = SqueezeExcitation()(x)
    x = SpatialAttention()(x)

    # Apply convolutional layers
    x = Conv2DSkip(x, 16, 3, activation='relu', padding='same')
    x = SqueezeExcitation()(x)
    x = Conv2DSkip(x, 16, 3, activation='relu', padding='same')

    x = BatchNormalization()(x)
    x = SoftThresholdLayer()(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)

    output = layers.Conv2D(1, 1, activation='linear', padding='same')(x)

    # Define the model
    model = Model(inputs, output, name='qr_correction_model')

    return model


@tf.function
@keras.saving.register_keras_serializable()
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
@keras.saving.register_keras_serializable()
def binarized_bce_loss(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_binary, y_pred_binary))


@tf.function
@keras.saving.register_keras_serializable()
def edge_loss(y_true, y_pred):
    y_true_edges = tf.image.sobel_edges(y_true)
    y_pred_edges = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))


@tf.function
@keras.saving.register_keras_serializable()
def loss_func(y_true, y_pred):
    # Define initial empirical loss values
    BCE_initial = 10.38
    MSE_initial = 0.768
    Edge_initial = 1.87

    # Compute the current loss values
    mse = mse_loss(y_true, y_pred)
    bce = binarized_bce_loss(y_true, y_pred)
    edge = edge_loss(y_true, y_pred)

    # Calculate the initial sum of losses
    L_initial = BCE_initial + MSE_initial + Edge_initial
    epsilon = tf.keras.backend.epsilon()  # Small constant

    # Prevent division by zero
    L_initial = tf.maximum(L_initial, epsilon)

    # Normalize each loss based on current values
    normalized_bce = (bce / L_initial) * BCE_initial
    normalized_mse = (mse / L_initial) * MSE_initial
    normalized_edge = (edge / L_initial) * Edge_initial

    # weights
    alpha = 0.5  # Weight for MSE
    gamma = 1  # Weight for BCE
    epsilon = 0.5  # Weight for Edge

    # Compute the composite loss
    composite_loss = alpha * normalized_mse + gamma * normalized_bce + epsilon * normalized_edge

    return composite_loss


def train_model(resolution=256, epochs=100, batch_size=64, jit=False):
    strategy = tf.distribute.MirroredStrategy()

    # Create the model
    input_shape = (resolution, resolution, 1)

    with strategy.scope():
        callbacks = [
            TensorBoard(log_dir="image_clean", histogram_freq=1, write_graph=True, write_images=False,
                        update_freq='epoch'),
            ModelCheckpoint(
                filepath='best_model.keras',  # Path to save the model
                monitor='loss',  # Metric to monitor
                save_best_only=True,  # Only save the best model
                mode='min',  # Save the model with the minimum loss
                verbose=1  # Print messages when the model is saved
            )
        ]

        optimizer = Adafactor(
            learning_rate=1.0,
            clipnorm=1.0
        )

        model = create_model(input_shape)

        dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size)

        if jit:
            jit = True
        else:
            jit = "auto"

        # Compile the model
        model.compile(optimizer=optimizer, loss="mse",
                      jit_compile=jit)

        model.summary()

        # Train the model
        model.fit(dataset, epochs=epochs, steps_per_epoch=250, callbacks=callbacks, verbose=1)

    # Save the model
    model.save('qr_correction_model.keras')


if __name__ == '__main__':
    # get batch size argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-JIT", nargs="?", default=False, const=True, type=bool,
                        help="Enable Just-In-Time compilation.")

    batch_size = parser.parse_args().batch_size
    jit_compile = parser.parse_args().JIT

    train_model(batch_size=batch_size, jit=jit_compile)
