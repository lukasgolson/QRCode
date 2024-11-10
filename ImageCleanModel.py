import argparse
import datetime
from os import mkdir
from pathlib import Path

import keras
import tensorflow as tf
from keras import Model
from keras.api import mixed_precision
from keras.src import layers
from keras.src.layers import Conv2D, Concatenate
from tensorflow.keras.callbacks import TensorBoard

import Dataset
from GanTrainingLoop import train_gan
from RollingAverageCheckpoint import RollingAverageModelCheckpoint
from layers.CoordConv import CoordConv
from layers.DeformableConv2D import DeformableConv2D
from layers.SqueezeExcitation import SqueezeExcitation

# Enable mixed precision

keras.config.set_dtype_policy("mixed_float16")


def Conv2DSkip(input_layer, filters, kernel_size, padding='same', coordConv=False):
    # Define the convolutional skip layer with Squeeze Excitation
    skip = input_layer
    if skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding)(skip)

    if coordConv:
        x = CoordConv(filters, kernel_size)(input_layer)
    else:
        x = layers.Conv2D(filters, kernel_size, padding=padding)(input_layer)

    x = layers.LeakyReLU()(x)
    x = SqueezeExcitation()(x)
    x = layers.Add()([skip, x])
    x = layers.LeakyReLU()(x)
    return x


def create_generator(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    localCnn = layers.Conv2D(8, 3, padding='same')(x)
    globalCnn = layers.Conv2D(8, 3, padding='same')(localCnn)
    x = Concatenate(axis=-1)([localCnn, globalCnn])

    x = DeformableConv2D(16, 3, 1)(x)
    x = layers.LeakyReLU()(x)
    x = DeformableConv2D(32, 3, 1)(x)
    x = layers.LeakyReLU()(x)
    x = Conv2DSkip(x, 48, 3, coordConv=True)
    x = Conv2DSkip(x, 64, 3, coordConv=False)
    x = Conv2DSkip(x, 96, 3, coordConv=False)
    x = Conv2DSkip(x, 112, 3, coordConv=False)
    output = Conv2D(1, 1, padding='same', activation='sigmoid', dtype='float32')(x)
    model = Model(inputs, output, name='qr_correction_model')
    return model


def create_discriminator(input_shape):
    dirty_inputs = layers.Input(shape=input_shape, name='dirty_input')
    clean_inputs = layers.Input(shape=input_shape, name='clean_input')
    inputs = layers.Concatenate(axis=-1)([dirty_inputs, clean_inputs])

    x = inputs

    x = Conv2DSkip(x, 16, 3, coordConv=True)

    for i in range(6):
        x = Conv2DSkip(x, 16 * (i + 1), 3, coordConv=False)
        x = Conv2D(16 * (i + 1), 3, strides=2, padding='same')(x)

    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(1, dtype='float32')(x)
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
