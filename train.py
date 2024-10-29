import datetime
import os

import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import Callback, LearningRateScheduler
from keras.src.layers import layer
from tensorflow.keras.callbacks import TensorBoard

from DataGenerator import QRDataGenerator
from Dataset import create_dataset
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


@keras.saving.register_keras_serializable()
def masked_categorical_crossentropy(y_true, y_pred):
    """
    Compute masked categorical crossentropy loss.

    :param y_true: Ground truth labels (one-hot encoded).
    :param y_pred: Predicted labels (output from the model).
    :return: Mean masked categorical crossentropy loss.
    """
    # Calculate categorical crossentropy
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

    # Create a mask to ignore fully 0 timesteps based on y_true
    # A timestep is considered valid if there's any non-zero value in that timestep
    mask = tf.reduce_sum(y_true, axis=-1) > 0  # Creates a boolean mask for valid timesteps

    # Convert mask to float (1 for valid, 0 for padded)
    mask = tf.cast(mask, tf.float32)

    # Apply the mask: multiply loss by the mask
    masked_loss = loss * mask

    # Return the mean loss, considering only valid timesteps
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)


def freeze_layer(model, layer_name, frozen=True, recompile=True):
    """Freeze or unfreeze the specified layer."""

    model_layer = model.get_layer(layer_name)

    layer.trainable = frozen

    if recompile:
        compile_model(model)

    print(f'Layer "{model_layer.name}" is {"frozen" if frozen else "unfrozen"}.')


def get_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    input_shape = (target_image_size, target_image_size, 1)  # Define the input shape for the images

    model = create_model(input_shape, max_sequence_length, num_chars)

    return model


def compile_model(model):
    # optimizer = tf.keras.optimizers.AdamW()
    optimizer = tf.keras.optimizers.Adafactor()

    model.compile(optimizer=optimizer, loss=masked_categorical_crossentropy,
                  metrics=['accuracy', 'precision', 'recall'], jit_compile=True)

    return model


def make_or_restore_model(max_sequence_length=512, num_chars=128, target_image_size=512, gradient_accumulation_steps=1,
                          compile=True):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        model = keras.models.load_model(latest_checkpoint, compile=False)
    else:
        print("Creating a new model")
        model = get_model(max_sequence_length, num_chars, target_image_size)

    if compile:
        return compile_model(model)

    return model


def create_lr_scheduler(initial_lr, max_lr, min_lr, warmup_epochs, period_epochs):
    """Creates a learning rate scheduler function with closed-over parameters."""

    def lr_scheduler(epoch, lr):
        """Calculate the learning rate based on the current epoch."""
        if epoch < warmup_epochs:
            # Linearly increase learning rate during warmup
            return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Calculate the cosine annealing phase
            cycle = np.floor(1 + (epoch - warmup_epochs) / period_epochs)
            x = np.abs((epoch - warmup_epochs) / period_epochs - 2 * cycle + 1)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * x))

    return keras.callbacks.LearningRateScheduler(lr_scheduler)


def run_training(epochs=24, headless_epochs=6, batch_size=16,  total_items_per_epoch=16 * 500,
                 gradient_accumulation_steps=None):
    max_sequence_length = 512
    num_chars = 128
    target_image_size = 512

    steps_per_epoch = total_items_per_epoch // batch_size

    assert epochs > 0, "The number of epochs should be greater than 0."

    assert epochs >= headless_epochs, "The number of epochs should be greater than or equal to the headless epochs."

    # keras.mixed_precision.set_global_policy("mixed_float16")

    strategy = tf.distribute.MirroredStrategy()

    tf.keras.backend.clear_session()

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # learning_scheduler = create_lr_scheduler(0.0001, 0.001, 0.0001, warming_epochs,
    #                                           period_epochs=int(epochs // periods))

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=500),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch"
        )
    ]

    dataset = create_dataset(target_size=(target_image_size, target_image_size),
                             batch_size=batch_size, shuffle=False, max_seq_len=max_sequence_length, num_chars=num_chars)

    print("Created data generator")

    # Save the model
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with strategy.scope():
        model = make_or_restore_model(max_sequence_length, num_chars, target_image_size, gradient_accumulation_steps,
                                      compile=False)
        model.summary()

        model.save(os.path.join(save_path, f'qr_model_empty_{date}.keras'))

        freeze_layer(model, 'spatial_transformer', frozen=True, recompile=True)

        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=headless_epochs,
                  callbacks=callbacks)

        freeze_layer(model, 'spatial_transformer', frozen=False, recompile=True)

        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                  callbacks=callbacks,
                  initial_epoch=headless_epochs)

    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))


if __name__ == "__main__":
    epochs = 48
    batch_size = 12
    run_training(epochs=epochs, periods=epochs // 8, batch_size=batch_size, gradient_accumulation_steps=None,
                 total_items_per_epoch=batch_size * 500)
    print("Training complete.")

# %%
