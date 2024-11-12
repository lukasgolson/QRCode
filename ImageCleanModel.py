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


def residual_block(x, filters, kernel_size=3, use_SE=True):
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
    if use_SE:
        x = SqueezeExcitation()(x)

    # Add skip connection
    x = Add()([skip, x])
    return x


def create_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Initial Encoder Layer with CoordConv
    x1 = CoordConv(16, 3, padding='same')(inputs)
    x1 = residual_block(x1, 16)
    x1 = DeformableConv2D(16, 3, 1)(x1)  # First deformable conv
    x1 = LeakyReLU()(x1)

    x2 = MaxPooling2D()(x1)  # Downscale
    x2 = Conv2D(32, 3, padding='same')(x2)
    x2 = residual_block(x2, 32, use_SE=True)
    x2 = DeformableConv2D(32, 3, 1)(x2)  # Second deformable conv
    x2 = LeakyReLU()(x2)

    x3 = MaxPooling2D()(x2)  # Downscale
    x3 = Conv2D(64, 3, padding='same')(x3)
    x3 = residual_block(x3, 64, use_SE=True)

    # Bottleneck with CoordConv and Deformable Conv
    x4 = MaxPooling2D()(x3)  # Downscale
    x4 = CoordConv(128, 3, padding='same')(x4)
    x4 = residual_block(x4, 128)
    x4 = DeformableConv2D(128, 3, 1)(x4)  # Bottleneck deformable conv
    x4 = LeakyReLU()(x4)

    # Decoder: Upsample to match encoder dimensions before concatenation
    x5 = UpSampling2D()(x4)  # Upscale to match x3
    x5 = Concatenate()([x5, x3])  # Skip connection
    x5 = Conv2D(64, 3, padding='same')(x5)
    x5 = residual_block(x5, 64, use_SE=True)

    x6 = UpSampling2D()(x5)  # Upscale to match x2
    x6 = Concatenate()([x6, x2])  # Skip connection
    x6 = Conv2D(32, 3, padding='same')(x6)
    x6 = residual_block(x6, 32, use_SE=True)

    x7 = UpSampling2D()(x6)  # Upscale to match x1
    x7 = Concatenate()([x7, x1])  # Skip connection
    x7 = Conv2D(16, 3, padding='same')(x7)
    x7 = residual_block(x7, 16)

    # Output layer with CoordConv
    x7 = CoordConv(16, 3, padding='same')(x7)  # CoordConv before output
    output = Conv2D(1, 1, padding='same', activation='sigmoid', dtype='float32')(x7)

    model = Model(inputs, output, name='qr_correction_model')
    return model


def create_discriminator(input_shape):
    # Define input layers for clean and dirty QR images
    dirty_inputs = Input(shape=input_shape, name='dirty_input')
    clean_inputs = Input(shape=input_shape, name='clean_input')
    inputs = Concatenate(axis=-1)([dirty_inputs, clean_inputs])

    # Initial CoordConv for spatial awareness
    x = CoordConv(16, 3, padding='same')(inputs)
    x = residual_block(x, 16)

    # Downsampling layers with depthwise separable convolutions
    for filters in [32, 64, 128]:
        x = DepthwiseConv2D(filters, strides=2, padding='same')(x)
        x = residual_block(x, filters, use_SE=True)

    # Additional Conv2D layers for feature extraction with reduced filters
    for filters in [128, 256]:
        x = residual_block(x, filters, use_SE=True)
        x = Conv2D(filters, 3, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

    # Global pooling and dense layers for binary classification
    x = GlobalMaxPooling2D()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(1, dtype='float32')(x)  # Output for discriminator

    model = Model(inputs=[dirty_inputs, clean_inputs], outputs=x, name='discriminator_model')
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
              steps_per_epoch=250, accumulation_steps=accumulation_steps, steps_per_log=10, disc_steps=1, lambda_l1=0.1,
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
