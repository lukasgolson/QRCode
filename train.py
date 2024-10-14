import datetime
import os

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import layers, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dropout

from DataGenerator import QRDataGenerator
from layers.SpatialTransformer import SpatialTransformerInputHead
from layers.involution import Involution

# Paths to directories
image_dir = 'data/images'
content_dir = 'data/contents'

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Data generator for images and text content


# Spatial Transformer Layer


gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")


@keras.saving.register_keras_serializable(package="qr_model", name="positional_encoding")
def positional_encoding(length, depth):
    """Generates a positional encoding matrix for a given sequence length and depth (embedding size)."""
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))

    # Apply sine to even indices and cosine to odd indices
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices in the array
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices in the array

    pos_encoding = pos_encoding[np.newaxis, ...]  # Add batch dimension
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_involution_architecture(input_tensor, length, channels=16, group_number=1):
    x = input_tensor



    for i in range(length):
        print(f"Involution layer {i}")
        channels_count = channels * (2 ** i)

        x = keras.layers.Conv2D(channels_count, (1, 1), activation='relu')(x)

        x, _ = Involution(
            channel=channels_count, group_number=group_number, kernel_size=3, stride=2, reduction_ratio=2)(x)

        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Instantiate the SpatialTransformerInputHead
    processing_head = SpatialTransformerInputHead()(inputs)  # Ensure the output is used correctly

    print(processing_head.shape)

    # Build the involution architecture

    x = processing_head

    x = create_involution_architecture(x, 4, 16, 4)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization
    x = layers.Dense(512, activation='relu')(x)

    x = Dropout(0.25)(x)

    # reduce to 512X512X1
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    # Flatten and reshape for sequence prediction
    sequence = layers.Flatten()(x)
    sequence = layers.Dense(max_sequence_length * num_chars, activation='relu')(sequence)
    sequence = layers.Reshape((max_sequence_length, -1))(sequence)  # Reshape to (sequence_length, feature_size)

    pos_encoding = positional_encoding(max_sequence_length, x.shape[-1])
    sequence += pos_encoding

    sequence = layers.Dense(512, activation='relu')(sequence)
    sequence = layers.BatchNormalization()(sequence)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(sequence)

    return Model(inputs, outputs, name='qr_model')


def get_compiled_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    input_shape = (target_image_size, target_image_size, 1)  # Define the input shape for the images

    model = create_model(input_shape, max_sequence_length, num_chars)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def make_or_restore_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(max_sequence_length, num_chars, target_image_size)

from char_level_encoder import CharLevelEncoder






def run_training(epochs=1, batch_size=16):
    max_sequence_length = 512
    num_chars = 128
    target_image_size = 512

    strategy = tf.distribute.MirroredStrategy()

    tf.keras.backend.clear_session()

    with strategy.scope():
        model = make_or_restore_model(max_sequence_length, num_chars, target_image_size)
        model.summary()

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [

        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch"
        )

    ]

    qr_data_gen = QRDataGenerator(image_dir, content_dir, batch_size=batch_size,
                                  max_sequence_length=max_sequence_length, num_chars=num_chars,
                                  target_size=(target_image_size, target_image_size))




    model.fit(qr_data_gen, epochs=epochs, callbacks=callbacks)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the model
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))


if __name__ == "__main__":
    run_training(epochs=16, batch_size=16)
