import argparse
import datetime
from os import mkdir
from pathlib import Path

import keras
import tensorflow as tf
from keras import Model, Input
from keras.api import mixed_precision
from keras.src import layers
from keras.src.layers import Conv2D, Concatenate, DepthwiseConv2D, UpSampling2D, LeakyReLU, MaxPooling2D, Add, \
    GlobalMaxPooling2D, Dense
from tensorflow.keras.callbacks import TensorBoard

import Dataset
from GanTrainingLoop import train_gan
from RollingAverageCheckpoint import RollingAverageModelCheckpoint
from layers.CoordConv import CoordConv
from layers.DeformableConv2D import DeformableConv2D
from layers.SqueezeExcitation import SqueezeExcitation

# Enable mixed precision

keras.config.set_dtype_policy("mixed_float16")


def residual_block(x, filters, kernel_size=3, use_squeeze_excitation=True):
    """ Residual block with optional Squeeze-and-Excitation and channel matching """
    skip = x
    # Adjust skip connection to match the number of filters if necessary
    if skip.shape[-1] != filters:
        skip = Conv2D(filters, (1, 1), padding='same')(skip)

    # Apply convolutions
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)

    # Optional Squeeze-and-Excitation
    if use_squeeze_excitation:
        x = SqueezeExcitation()(x)

    # Add skip connection
    x = Add()([skip, x])
    return x

def create_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Initial Encoder Layer with CoordConv
    x1 = CoordConv(32, 3, padding='same')(inputs)  # Increased initial filters
    x1 = residual_block(x1, 32)
    x1 = DeformableConv2D(32, 3, 1)(x1)
    x1 = LeakyReLU()(x1)

    x2 = MaxPooling2D()(x1)
    x2 = Conv2D(64, 3, padding='same')(x2)  # Increased filters in encoder
    x2 = residual_block(x2, 64, use_squeeze_excitation=True)
    x2 = DeformableConv2D(64, 3, 1)(x2)
    x2 = LeakyReLU()(x2)

    x3 = MaxPooling2D()(x2)
    x3 = Conv2D(128, 3, padding='same')(x3)  # Increased filters
    x3 = residual_block(x3, 128, use_squeeze_excitation=True)

    # Enhanced Bottleneck with higher filters and additional layer
    x4 = MaxPooling2D()(x3)
    x4 = CoordConv(192, 3, padding='same')(x4)  # Higher bottleneck filters
    x4 = residual_block(x4, 192)
    x4 = DeformableConv2D(192, 3, 1)(x4)
    x4 = Conv2D(192, 3, padding='same')(x4)  # Additional Conv layer
    x4 = LeakyReLU()(x4)

    # Decoder with higher filters
    x5 = UpSampling2D()(x4)
    x5 = Concatenate()([x5, x3])
    x5 = Conv2D(128, 3, padding='same')(x5)
    x5 = residual_block(x5, 128, use_squeeze_excitation=True)

    x6 = UpSampling2D()(x5)
    x6 = Concatenate()([x6, x2])
    x6 = Conv2D(64, 3, padding='same')(x6)
    x6 = residual_block(x6, 64, use_squeeze_excitation=True)

    x7 = UpSampling2D()(x6)
    x7 = Concatenate()([x7, x1])
    x7 = Conv2D(32, 3, padding='same')(x7)
    x7 = residual_block(x7, 32)

    # Output layer
    x7 = CoordConv(32, 3, padding='same')(x7)

    x7 = LeakyReLU()(x7)

    x8 = Conv2D(16, 2, padding='same')(x7)
    x8 = LeakyReLU()(x8)

    output = Conv2D(1, 1, padding='same', activation='sigmoid', dtype='float32')(x8)

    model = Model(inputs, output, name='enhanced_generator_model')
    return model


def create_discriminator(input_shape):
    dirty_inputs = Input(shape=input_shape, name='dirty_input')
    clean_inputs = Input(shape=input_shape, name='clean_input')
    inputs = Concatenate(axis=-1)([dirty_inputs, clean_inputs])

    # Initial CoordConv for spatial awareness
    x = CoordConv(16, 3, padding='same')(inputs)

    # Reduced filter sizes in deeper layers to decrease complexity
    for filters in [16, 32, 48]:  # Reduced filters from 16, 32, 64
        x = DepthwiseConv2D(kernel_size=3, padding='same', strides=2)(x)
        x = Conv2D(filters, kernel_size=1, padding='same')(x)
        x = residual_block(x, filters)  # Removed SE layers in initial layers
        x = LeakyReLU()(x)

    # Further reduce filters and remove SE block in final layers
    for filters in [64, 128]:  # Reduced from 128, 256
        x = Conv2D(filters, 3, strides=2, padding='same')(x)
        x = residual_block(x, filters, use_squeeze_excitation=True)  # Remove SE here to reduce complexity
        x = LeakyReLU()(x)

    # Global pooling and dense layers for binary classification
    x = GlobalMaxPooling2D()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dense(1, dtype='float32')(x)

    model = Model(inputs=[dirty_inputs, clean_inputs], outputs=x, name='simplified_discriminator_model')
    return model


def train_model(resolution=256, epochs=100, batch_size=32, accumulation_steps=4, resume=False,
                checkpoint_dir='checkpoints'):
    generator = create_generator((resolution, resolution, 1))
    if not Path("models").exists():
        mkdir("models")
        print("Directory 'models' created")
    generator.save("models/qr_correction_model.keras")
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

    # Wrap optimizers with LossScaleOptimizer for mixed precision
    gen_optimizer = mixed_precision.LossScaleOptimizer(gen_optimizer)
    disc_optimizer = mixed_precision.LossScaleOptimizer(disc_optimizer)

    train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataset, val_dataset, epochs, callbacks,
              steps_per_epoch=250, accumulation_steps=accumulation_steps, steps_per_log=10, disc_steps=1,
              resume=resume, checkpoint_dir=checkpoint_dir)

    generator.save('qr_correction_model.keras')
    discriminator.save('discriminator_model.keras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=4)

    parser.add_argument("-resume", nargs="?", default=False, const=True, type=bool,
                        help="Resume training from a checkpoint.")
    parser.add_argument("-checkpoint_dir", type=str, default='checkpoints', help="Directory to save training state.")

    batch_size = parser.parse_args().batch_size
    accumulation = parser.parse_args().accumulation_steps

    resume = parser.parse_args().resume
    checkpoint_dir = parser.parse_args().checkpoint_dir

    train_model(batch_size=batch_size, accumulation_steps=accumulation, resume=resume, checkpoint_dir=checkpoint_dir)
