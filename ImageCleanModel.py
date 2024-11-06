import argparse
import datetime

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
from RollingAverageCheckpoint import RollingAverageModelCheckpoint
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


def create_generator(input_shape):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Apply spatial transformer to the input image
    x = SpatialTransformer()(inputs)
    x = SqueezeExcitation()(x)
    x = SpatialAttention()(x)
    x = BatchNormalization()(x)

    x = Conv2DSkip(x, 32, 3, activation='relu', padding='same')
    x = SqueezeExcitation()(x)
    x = Conv2DSkip(x, 16, 3, activation='relu', padding='same')

    x = BatchNormalization()(x)
    # x = SoftThresholdLayer()(x)

    x = Conv2DSkip(x, 8, 3, activation='relu', padding='same')

    output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    # Define the model
    model = Model(inputs, output, name='qr_correction_model')

    return model


def create_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

    model = Model(inputs, x, name='discriminator_model')
    return model


@tf.function
@keras.saving.register_keras_serializable()
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_gan(generator, discriminator, gen_optimizer, adv_optimizer, dataset, val_dataset, epochs, batch_size,
              resolution, callbacks, jit_compile="auto"):
    # Compile the discriminator for its own training
    discriminator.compile(optimizer=adv_optimizer, loss='binary_crossentropy',
                          jit_compile=jit_compile)

    # GAN Model (discriminator initially not trainable for generator updates)
    gan_input = layers.Input(shape=(resolution, resolution, 1))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(optimizer=gen_optimizer, loss='binary_crossentropy', jit_compile=jit_compile)

    for callback in callbacks:
        callback.set_model(gan_model)

    g_loss = float('inf')
    d_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step, (clean_images, dirty_images) in enumerate(dataset):
            # Generate transformed images
            transformed_images = generator.predict(dirty_images)

            # Concatenate and shuffle
            combined_images = tf.concat([clean_images, transformed_images], axis=0)
            combined_labels = tf.concat([tf.ones((batch_size, 1)) * 0.9, tf.zeros((batch_size, 1))],
                                        axis=0)  # Label smoothing

            indices = tf.range(start=0, limit=tf.shape(combined_images)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_images = tf.gather(combined_images, shuffled_indices)
            shuffled_labels = tf.gather(combined_labels, shuffled_indices)

            # Split shuffled data back into batches of size batch_size
            num_batches = 2
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                d_loss = discriminator.train_on_batch(shuffled_images[start:end], shuffled_labels[start:end])

            # Train Generator
            discriminator.trainable = False  # Freeze discriminator during generator training
            g_loss = gan_model.train_on_batch(dirty_images, tf.ones((batch_size, 1)))

            print(f"Step: {step + 1}/{len(dataset)}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Validation
        val_mse = 0
        val_step = 0
        for val_step, (val_real_images, val_dirty_images) in enumerate(val_dataset):
            val_fake_images = generator.predict(val_dirty_images)
            val_mse += tf.reduce_mean(tf.square(val_real_images - val_fake_images)).numpy()

        val_mse /= (val_step + 1)
        print(f"Validation MSE after Epoch {epoch + 1}: {val_mse:.4f}")

        # Trigger callbacks
        logs = {'loss': g_loss, 'val_mse': val_mse}
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    for callback in callbacks:
        callback.on_train_end(None)


def train_model(resolution=256, epochs=100, batch_size=32, jit=False):
    strategy = tf.distribute.MirroredStrategy()

    # Create the model
    input_shape = (resolution, resolution, 1)

    log_dir = "logs/fit/ImageCleanModel/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with strategy.scope():

        generator = create_generator(input_shape)
        discriminator = create_discriminator(input_shape)

        dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size)
        val_dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size)

        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False,
                        update_freq='epoch'),
            RollingAverageModelCheckpoint(
                filepath='best_model.keras',
                monitor='loss',
                save_best_only=True,
                mode='min',
                verbose=1,
                rolling_epochs=10
            )
        ]

        gen_optimizer = Adafactor(
            learning_rate=1.0,
            clipnorm=1.0
        )

        adv_optimizer = Adafactor(
            learning_rate=1.0,
            clipnorm=1.0
        )

        if jit:
            jit = True
        else:
            jit = "auto"

        # Train the model
        train_gan(generator, discriminator, gen_optimizer, adv_optimizer, dataset, val_dataset, epochs, batch_size,
                  resolution, callbacks, jit_compile=jit)

    # Save the model
    generator.save('qr_correction_model.keras')


if __name__ == '__main__':
    # get batch size argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-JIT", nargs="?", default=False, const=True, type=bool,
                        help="Enable Just-In-Time compilation.")

    batch_size = parser.parse_args().batch_size
    jit_compile = parser.parse_args().JIT

    train_model(batch_size=batch_size, jit=jit_compile)
