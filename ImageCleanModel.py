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
from keras.src.trainers.compile_utils import CompileLoss
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
def train_gan(generator, discriminator, gen_optimizer, adv_optimizer, dataset, val_dataset, epochs, callbacks, steps_per_epoch=250):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Limit the number of steps per epoch if specified
        steps_in_epoch = steps_per_epoch or len(dataset)

        for step, (clean_images, dirty_images) in enumerate(dataset.take(steps_in_epoch)):  # Use .take to limit steps
            # Generate transformed images
            transformed_images = generator(dirty_images, training=True)

            # Train Discriminator manually
            with tf.GradientTape() as tape_d:
                # Apply label smoothing
                real_labels = tf.ones_like(real_pred) * 0.9  # Smooth real labels to 0.9 instead of 1
                fake_labels = tf.zeros_like(fake_pred) + 0.1  # Smooth fake labels to 0.1 instead of 0

                # Get predictions from the discriminator
                real_pred = discriminator(clean_images, training=True)
                fake_pred = discriminator(transformed_images, training=True)

                # Discriminator loss (binary cross-entropy)
                d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)

                # Combine the losses
                d_loss = (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)) / 2

            # Get discriminator gradients and apply them
            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            adv_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

            # Train Generator (via GAN model) manually
            with tf.GradientTape() as tape_g:
                generated_images = generator(dirty_images, training=True)
                gan_output = discriminator(generated_images, training=True)
                # Generator loss is based on how well the generated images fool the discriminator
                g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(gan_output), gan_output))

            # Get generator gradients and apply them
            grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

            if step % 10 == 0:
                print(f"Step: {step + 1}/{steps_in_epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Validation step
        val_mse = 0
        val_step = 0  # Initialize the val_step variable
        for val_step, (val_real_images, val_dirty_images) in enumerate(val_dataset):
            val_fake_images = generator(val_dirty_images, training=False)
            # Accumulate the squared differences (MSE)
            val_mse += tf.reduce_mean(tf.square(val_real_images - val_fake_images)).numpy()

        # Average MSE over all validation steps
        val_mse /= (val_step + 1)  # `val_step` is the index of the last batch, so +1 for the total number of steps
        print(f"Validation MSE after Epoch {epoch + 1}: {val_mse:.4f}")

        # Trigger callbacks
        logs = {'loss': g_loss, 'val_mse': val_mse}
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    for callback in callbacks:
        callback.on_train_end(None)



def train_model(resolution=256, epochs=100, batch_size=32, jit=False):
    strategy = tf.distribute.MirroredStrategy()

    #with strategy.scope():
    generator = create_generator((resolution, resolution, 1))
    discriminator = create_discriminator((resolution, resolution, 1))

    dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size)
    val_dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size)

    callbacks = [
        TensorBoard(log_dir="logs/fit/ImageCleanModel/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        RollingAverageModelCheckpoint(filepath='best_model.keras', monitor='loss', save_best_only=True, mode='min')
    ]

    gen_optimizer = Adafactor(learning_rate=1.0, clipnorm=1.0)
    adv_optimizer = Adafactor(learning_rate=0.01, clipnorm=1.0)

    jit = jit if jit else "auto"

    train_gan(generator, discriminator, gen_optimizer, adv_optimizer, dataset, val_dataset, epochs, callbacks)

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
