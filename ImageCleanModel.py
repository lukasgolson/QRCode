import keras
from keras import Model
from keras.src import layers

import tensorflow as tf
from keras import backend as K
from keras.src.optimizers import Adafactor
from tensorflow.keras.callbacks import TensorBoard

import Dataset
from layers.SpatialAttention import SpatialAttention
from layers.SpatialTransformer import SpatialTransformer

def encoder(x):
    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def decoder(x):
    for filters in [256, 128, 64]:
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    return x

def create_model(input_shape):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Apply spatial transformer to the input image
    x = SpatialTransformer()(inputs)


    x = encoder(x)

    x = SpatialAttention()(x)


    x = decoder(x)



    # Define the model
    model = Model(inputs, x, name='qr_correction_model')

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


def train_model(resolution=256, epochs=100):
    strategy = tf.distribute.MirroredStrategy()

    # Create the model
    input_shape = (resolution, resolution, 1)


    callbacks = [
        TensorBoard(log_dir="image_clean", histogram_freq=1, write_graph=True, write_images=False, update_freq='epoch')
    ]

    optimizer = Adafactor(
        learning_rate=1.0
    )


    with strategy.scope():
        model = create_model(input_shape)

        dataset = Dataset.create_dataset(paired=True, target_size=(resolution, resolution))

        # Compile the model
        model.compile(optimizer=optimizer, loss=combined_mse_ssim_loss, metrics=[ssim_loss])

        model.summary()

    # Train the model
    model.fit(dataset, epochs=epochs, steps_per_epoch=250, callbacks=callbacks)

    # Save the model
    model.save('qr_correction_model.keras')


if __name__ == '__main__':
    train_model()
