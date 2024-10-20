import datetime
import os

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from DataGenerator import QRDataGenerator
from Model import create_model

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


def get_compiled_model(max_sequence_length=512, num_chars=128, target_image_size=512, gradient_accumulation_steps=1):
    input_shape = (target_image_size, target_image_size, 1)  # Define the input shape for the images

    model = create_model(input_shape, max_sequence_length, num_chars)
    model.compile(optimizer=keras.optimizers.AdamW(gradient_accumulation_steps=gradient_accumulation_steps),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def make_or_restore_model(max_sequence_length=512, num_chars=128, target_image_size=512, gradient_accumulation_steps=1):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(max_sequence_length, num_chars, target_image_size, gradient_accumulation_steps)


def run_training(epochs=1, batch_size=16, gradient_accumulation_steps=None):
    max_sequence_length = 512
    num_chars = 128
    target_image_size = 512

    strategy = tf.distribute.MirroredStrategy()

    keras.mixed_precision.set_global_policy("mixed_float16")

    tf.keras.backend.clear_session()

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [

        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq=500),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch"
        )

    ]

    qr_data_gen = QRDataGenerator(image_dir, content_dir, batch_size=batch_size,
                                  max_sequence_length=max_sequence_length, num_chars=num_chars,
                                  target_size=(target_image_size, target_image_size))

    print("Created data generator")

    # Save the model
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with strategy.scope():
        model = make_or_restore_model(max_sequence_length, num_chars, target_image_size, gradient_accumulation_steps)
        model.summary()
        # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True, show_trainable=True, show_layer_activations=True, dpi=800)

        model.save(os.path.join(save_path, f'qr_model_empty_{date}.keras'))

        model.fit(qr_data_gen, epochs=epochs, callbacks=callbacks)

    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))


if __name__ == "__main__":
    run_training(epochs=8, batch_size=8, gradient_accumulation_steps=None)
    print("Training complete.")
