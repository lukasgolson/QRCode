import datetime
import os
import argparse

import tensorflow


import keras
import numpy as np
import tensorflow as tf
from keras.src.layers import layer
from tensorflow.keras.callbacks import TensorBoard

from DataGenerator import QRDataGenerator
from Dataset import create_dataset
from Model import create_model

tensorflow.keras.backend.clear_session(
    free_memory=True
)

# Paths to directories
image_dir = 'data/images'
content_dir = 'data/contents'

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Spatial Transformer Layer and GPUs configuration
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")


# @tf.function
@keras.saving.register_keras_serializable()
def masked_categorical_crossentropy(y_true, y_pred):
    # Compute masked categorical crossentropy loss.
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    mask = tf.cast(mask, tf.float32)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)


def freeze_layer(model, layer_name, frozen=True, recompile=True):
    model_layer = model.get_layer(layer_name)
    layer.trainable = frozen
    if recompile:
        compile_model(model)
    print(f'Layer "{model_layer.name}" is {"frozen" if frozen else "unfrozen"}.')


def get_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    input_shape = (target_image_size, target_image_size, 1)
    model = create_model(input_shape, max_sequence_length, num_chars)
    return model


def compile_model(model):

    if JIT_COMPILE:
        jit = True
    else:
        jit = "auto"

    optimizer = tf.keras.optimizers.Adafactor(clipnorm=1.0, learning_rate=1.0, weight_decay=0.0001)
    model.compile(optimizer=optimizer, loss=masked_categorical_crossentropy,
                  metrics=['accuracy'], jit_compile=jit, run_eagerly=RUN_EAGERLY)
    return model


def make_or_restore_model(max_sequence_length=512, num_chars=128, target_image_size=512, gradient_accumulation_steps=1,
                          compile=True):
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


def run_training(epochs, headless_epochs=6, batch_size=16, total_items_per_epoch=16 * 500,
                 gradient_accumulation_steps=None, max_sequence_length=512
                 , num_chars=128, target_image_size=512, built_in_generator=True):
    print("Running training with the following parameters:")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total items per epoch: {total_items_per_epoch}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Max sequence length: {max_sequence_length}")
    print(f"Number of characters: {num_chars}")
    print(f"Target image size: {target_image_size}")
    print(f"JIT compilation: {JIT_COMPILE}")
    print(f"Eager execution: {RUN_EAGERLY}")

    steps_per_epoch = total_items_per_epoch // batch_size
    strategy = tf.distribute.MirroredStrategy()
    tf.keras.backend.clear_session()
    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=500),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch")
    ]

    if built_in_generator:
        dataset = create_dataset(target_size=(target_image_size, target_image_size),
                                 batch_size=batch_size, shuffle=False, max_seq_len=max_sequence_length,
                                 num_chars=num_chars)
    else:
        dataset = QRDataGenerator(batch_size, max_sequence_length, num_chars, target_image_size)
    print("Created data generator")
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with strategy.scope():
        model = make_or_restore_model(max_sequence_length, num_chars, target_image_size, gradient_accumulation_steps,
                                      compile=False)
        model.summary()
        model.save(os.path.join(save_path, f'qr_model_empty_{date}.keras'))
        freeze_layer(model, 'spatial_transformer', frozen=True, recompile=True)
        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=headless_epochs, callbacks=callbacks, verbose=2)
        freeze_layer(model, 'spatial_transformer', frozen=False, recompile=True)
        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks,
                  initial_epoch=headless_epochs, verbose=2)

    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))


JIT_COMPILE = False
RUN_EAGERLY = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QR Code model training.")
    parser.add_argument("--epochs", type=int, default=48, help="Number of epochs to run training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Number of steps to accumulate gradients.")
    parser.add_argument("--target_image_size", type=int, default=512, help="Size of the target image.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("-JIT", nargs="?", default=False, const=True, type=bool,
                        help="Enable Just-In-Time compilation.")
    parser.add_argument("-EAGER", nargs="?", default=False, const=True, type=bool,
                        help="Enable eager run-time.")

    parser.add_argument("-EXTERN_DATA", nargs="?", default=False, const=True, type=bool,
                        help="Use external data generator.")

    JIT_COMPILE = parser.parse_args().JIT
    RUN_EAGERLY = parser.parse_args().EAGER

    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size,
                 gradient_accumulation_steps=args.gradient_accumulation_steps,
                 total_items_per_epoch=args.batch_size * 500, max_sequence_length=args.max_sequence_length,
                 target_image_size=args.target_image_size, built_in_generator=True)
    print("Training complete.")
