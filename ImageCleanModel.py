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
from layers.FoveatedConvolution import FoveatedConvolutionLayer
from layers.SoftThresholdLayer import SoftThresholdLayer
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformer
from layers.SqueezeExcitation import SqueezeExcitation


def Conv2DSkip(input_layer, filters, kernel_size, activation='relu', padding='same'):
    # Perform convolution on the input layer

    # Create a skip connection
    skip = input_layer

    x = layers.Conv2D(filters, kernel_size, padding=padding)(input_layer)
    x = layers.Activation(activation)(x)

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

    x = SpatialAttention(num_layers=4, initial_filters=8, filter_step=8)(inputs)

    x = SpatialTransformer(add_residual=False, trainable_residual=False)(x)

    # foveated cnn helps to focus on the important parts of the image;
    # helping spatial transformer to focus on the right
    x = FoveatedConvolutionLayer()(x)
    x = SqueezeExcitation()(x)
    x = BatchNormalization()(x)

    x = Conv2DSkip(x, 32, 3, padding='same')
    x = layers.LeakyReLU()(x)

    x = Conv2DSkip(x, 16, 3, padding='same')
    x = layers.LeakyReLU()(x)

    x = SqueezeExcitation()(x)

    x = SpatialAttention(num_layers=4, initial_filters=8, filter_step=8)(x)

    x = Conv2DSkip(x, 16, 3, padding='same')
    x = layers.LeakyReLU()(x)

    x = Conv2DSkip(x, 8, 3, padding='same')
    x = layers.LeakyReLU()(x)

    x = BatchNormalization()(x)

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


def train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataset, val_dataset, epochs, callbacks,
              log_interval=10, steps_per_epoch=250, steps_per_val=10, lambda_l1=0.1):
    # turn callbacks into a list
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    print("Callbacks: ", callbacks)

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks, add_history=True, model=generator)

    logs = {}
    callback_list.on_train_begin(logs=logs)

    generator.compile(optimizer=gen_optimizer, loss='binary_crossentropy')
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy')

    generator.summary()
    discriminator.summary()

    for epoch in range(epochs):
        callback_list.on_epoch_begin(epoch, logs=logs)

        print(f"Epoch {epoch + 1}/{epochs}")

        # Limit the number of steps per epoch if specified
        steps_in_epoch = steps_per_epoch or len(dataset)
        steps_in_val = steps_per_val or len(val_dataset)

        for step, (clean_images, dirty_images) in enumerate(dataset.take(steps_in_epoch)):  # Use .take to limit steps
            # Generate transformed images

            callback_list.on_batch_begin(step, logs=logs)
            callback_list.on_train_batch_begin(step, logs=logs)

            transformed_images = generator(dirty_images, training=True)

            # Train Discriminator manually
            with tf.GradientTape() as tape_d:
                real_pred = discriminator(clean_images, training=True)
                fake_pred = discriminator(transformed_images, training=True)

                # Apply label smoothing
                real_labels = tf.ones_like(real_pred) * 0.9  # Smooth real labels to 0.9 instead of 1
                fake_labels = tf.zeros_like(fake_pred) + 0.1  # Smooth fake labels to 0.1 instead of 0

                # Discriminator loss (binary cross-entropy)
                d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)

                # Combine the losses
                d_loss = (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)) / 2

            # Get discriminator gradients and apply them
            grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

            # Train Generator (via GAN model) manually
            with tf.GradientTape() as tape_g:
                generated_images = generator(dirty_images, training=True)
                gan_output = discriminator(generated_images, training=True)
                # Generator loss is based on how well the generated images fool the discriminator
                g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(gan_output),
                                                                            gan_output)) + lambda_l1 * tf.reduce_mean(
                    tf.abs(clean_images - generated_images))

            # Get generator gradients and apply them
            grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

            # Trigger callbacks

            logs = {'d_loss': d_loss, 'g_loss': g_loss}

            callback_list.on_train_batch_end(step, logs=logs)

            if step % log_interval == 0:
                print(f"Step: {step}/{steps_in_epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        # Validation step
        val_mse = 0
        val_step = 0  # Initialize the val_step variable
        for val_step, (val_real_images, val_dirty_images) in enumerate(dataset.take(steps_in_val)):
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

    gen_optimizer = Adafactor(learning_rate=1.0, clipnorm=1.0)
    adv_optimizer = Adafactor(learning_rate=0.01, clipnorm=1.0)

    jit = jit if jit else "auto"

    train_gan(generator, discriminator, gen_optimizer, adv_optimizer, dataset, val_dataset, epochs, callbacks,
              steps_per_epoch=250, steps_per_val=10)

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
