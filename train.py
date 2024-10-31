import datetime
import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

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

# Spatial Transformer Layer and GPUs configuration
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")


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
    optimizer = tf.keras.optimizers.Adafactor()
    model.compile(optimizer=optimizer, loss=masked_categorical_crossentropy,
                  metrics=['accuracy', 'precision', 'recall'], jit_compile=JIT_COMPILE)
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


def create_lr_scheduler(initial_lr, max_lr, min_lr, warmup_epochs, period_epochs):
    def lr_scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            cycle = np.floor(1 + (epoch - warmup_epochs) / period_epochs)
            x = np.abs((epoch - warmup_epochs) / period_epochs - 2 * cycle + 1)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * x))

    return keras.callbacks.LearningRateScheduler(lr_scheduler)


def run_training(epochs, headless_epochs=6, batch_size=16, total_items_per_epoch=16 * 500,
                 gradient_accumulation_steps=None, max_sequence_length=512
                 , num_chars=128, target_image_size=512):
    steps_per_epoch = total_items_per_epoch // batch_size
    strategy = tf.distribute.MirroredStrategy()
    tf.keras.backend.clear_session()
    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=500),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch")
    ]

    dataset = create_dataset(target_size=(target_image_size, target_image_size),
                             batch_size=batch_size, shuffle=False, max_seq_len=max_sequence_length, num_chars=num_chars)
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
        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=headless_epochs, callbacks=callbacks)
        freeze_layer(model, 'spatial_transformer', frozen=False, recompile=True)
        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks,
                  initial_epoch=headless_epochs)

    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QR Code model training.")
    parser.add_argument("--epochs", type=int, default=48, help="Number of epochs to run training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of steps to accumulate gradients.")
    parser.add_argument("--target_image_size", type=int, default=512, help="Size of the target image.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--JIT", type=bool, default=False, help="Enable Just-In-Time compilation.")

    global JIT_COMPILE
    JIT_COMPILE = parser.parse_args().JIT

    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size, gradient_accumulation_steps=None,
                 total_items_per_epoch=args.batch_size * 500, max_sequence_length=args.max_sequence_length, target_image_size=args.target_image_size)
    print("Training complete.")
