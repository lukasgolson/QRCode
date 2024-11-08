import argparse
import datetime
from os import mkdir
from pathlib import Path

import keras
import tensorflow as tf
from keras import Model
from keras.src import layers
from keras.src.layers import BatchNormalization, LayerNormalization
from keras.src.optimizers import Adafactor
from tensorflow.keras.callbacks import TensorBoard

import Dataset
from RollingAverageCheckpoint import RollingAverageModelCheckpoint
from layers.CoordConv import CoordConv
from layers.DeformableConv2D import DeformableConv2D
from layers.FoveatedConvolution import FoveatedConvolutionLayer
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformer
from layers.SqueezeExcitation import SqueezeExcitation


def Conv2DSkip(input_layer, filters, kernel_size, padding='same', coordConv=False):
    # Perform convolution on the input layer

    # Create a skip connection
    skip = input_layer

    if skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(skip)

    if coordConv:
        x = CoordConv(filters, kernel_size)(input_layer)
    else:
        x = layers.Conv2D(filters, kernel_size, padding=padding)(input_layer)

    x = layers.LeakyReLU()(x)

    x = SqueezeExcitation()(x)

    # Add the skip connection to the output of the convolutional layer
    x = layers.Add()([skip, x])
    x = layers.LeakyReLU()(x)

    # Apply the activation function after the addition
    return x


def create_generator(input_shape):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    x = inputs
    x = Conv2DSkip(x, 16, 3, padding='same')
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample by a factor of 2

    x = Conv2DSkip(x, 32, 3, padding='same')
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample by a factor of 2

    x = Conv2DSkip(x, 64, 3, padding='same', coordConv=True)

    x = LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = DeformableConv2D(64, 3)(x)

    x = LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D(size=(2, 2))(x)  # Upsample by a factor of 2
    x = Conv2DSkip(x, 64, 3, padding='same')

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv2DSkip(x, 32, 3, padding='same')

    x = FoveatedConvolutionLayer(fovea_size=(64, 64))(x)
    x = layers.LeakyReLU()(x)

    x = SqueezeExcitation(use_residual=False)(x)

    x = Conv2DSkip(x, 16, 3, padding='same')

    x = LayerNormalization()(x)

    output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    # Define the model
    model = Model(inputs, output, name='qr_correction_model')

    return model


def create_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    x = CoordConv(32, 3)(x)

    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)

    x = layers.LeakyReLU()(x)

    x = DeformableConv2D(64, 3)(x)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.concatenate([gap, gmp])

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dense(1)(x)  # Output layer for binary classification

    model = Model(inputs, x, name='discriminator_model')
    return model


@tf.function
@keras.saving.register_keras_serializable()
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def gradient_penalty(critic, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = real_images + epsilon * (fake_images - real_images)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((grad_l2 - 1.0) ** 2)

    return gp


def train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataset, val_dataset, epochs, callbacks,
              log_interval=10, steps_per_epoch=250, steps_per_val=10, disc_steps=3, lambda_l1=0.1, lambda_gp=10.0):
    # turn callbacks into a list
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    print("Callbacks: ", callbacks)

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks, add_history=True, model=generator)

    logs = {}
    callback_list.on_train_begin(logs=logs)

    generator.compile(optimizer=gen_optimizer)
    discriminator.compile(optimizer=disc_optimizer)

    generator.summary()
    discriminator.summary()

    for epoch in range(epochs):
        callback_list.on_epoch_begin(epoch, logs=logs)

        print(f"Epoch {epoch + 1}/{epochs}")

        # Limit the number of steps per epoch if specified
        steps_in_epoch = steps_per_epoch or len(dataset)
        steps_in_val = steps_per_val or len(val_dataset)

        disc_step_count = 0

        g_loss = tf.constant(-1.0)


        for step, (clean_images, dirty_images) in enumerate(dataset.take(steps_in_epoch)):  # Use .take to limit steps
            # Generate transformed images

            callback_list.on_batch_begin(step, logs=logs)
            callback_list.on_train_batch_begin(step, logs=logs)

            transformed_images = generator(dirty_images, training=True)

            # Train Discriminator manually
            with tf.GradientTape() as tape_d:
                real_pred = discriminator(clean_images, training=True)
                fake_pred = discriminator(transformed_images, training=True)

                identity_loss = sum(discriminator.losses)

                gp = gradient_penalty(discriminator, clean_images, transformed_images)

                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + identity_loss + lambda_gp * gp

            # Get discriminator gradients and apply them
            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

            disc_step_count += 1
            if disc_step_count >= disc_steps:
                disc_step_count = 0

                # Train Generator (via GAN model) manually
                with tf.GradientTape() as tape_g:
                    generated_images = generator(dirty_images, training=True)
                    gan_output = discriminator(generated_images, training=True)
                    identity_loss = sum(generator.losses)
                    g_loss = -tf.reduce_mean(gan_output) + lambda_l1 * tf.reduce_mean(
                        tf.abs(clean_images - generated_images)) + identity_loss

                # Get generator gradients and apply them
                grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

            else:
                g_loss = g_loss

            # Trigger callbacks

            logs = {'d_loss': d_loss, 'g_loss': g_loss}

            callback_list.on_train_batch_end(step, logs=logs)

            if step % log_interval == 0:
                print(f"Step: {step + log_interval}/{steps_in_epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Validation step
        val_mse = 0
        val_step = 0  # Initialize the val_step variable
        for val_step, (val_real_images, val_dirty_images) in enumerate(val_dataset.take(steps_in_val)):
            callback_list.on_test_batch_begin(val_step, logs=logs)

            val_fake_images = generator(val_dirty_images, training=False)
            # Accumulate the squared differences (MSE)
            val_mse += tf.reduce_mean(tf.square(val_real_images - val_fake_images)).numpy()

            callback_list.on_test_batch_end(val_step, logs=logs)

        # Average MSE over all validation steps
        val_mse /= (val_step + 1)  # `val_step` is the index of the last batch, so +1 for the total number of steps
        print(f"Validation MSE after Epoch {epoch + 1}: {val_mse:.4f}")

        # Trigger callbacks
        logs = {'d_loss': d_loss, 'g_loss': g_loss, 'val_mse': val_mse}
        callback_list.on_epoch_end(epoch, logs=logs)

    callback_list.on_train_end(logs=logs)


def train_model(resolution=256, epochs=100, batch_size=32, jit=False):
    strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():
    generator = create_generator((resolution, resolution, 1))

    if not Path("models").exists():
        mkdir("models")
        print("Directory 'models' created")
    generator.save("models/qr_correction_model.keras")

    # create models directory if it doesn't exist

    discriminator = create_discriminator((resolution, resolution, 1))

    dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size,
                                     noisiest_epoch=0)
    val_dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution), batch_size=batch_size,
                                         noisiest_epoch=50,
                                         batches_per_epoch=1)

    callbacks = [
        TensorBoard(log_dir="logs/fit/ImageCleanModel/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    profile_batch=0),
        RollingAverageModelCheckpoint(filepath='best_model.keras', monitor='val_mse', save_best_only=True, mode='min')
    ]

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

    jit = jit if jit else "auto"

    train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataset, val_dataset, epochs, callbacks,
              steps_per_epoch=250, steps_per_val=10, lambda_l1=0.1)

    generator.save('qr_correction_model.keras')
    discriminator.save('discriminator_model.keras')


if __name__ == '__main__':
    # get batch size argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-JIT", nargs="?", default=False, const=True, type=bool,
                        help="Enable Just-In-Time compilation.")

    batch_size = parser.parse_args().batch_size
    jit_compile = parser.parse_args().JIT

    train_model(batch_size=batch_size, jit=jit_compile)
