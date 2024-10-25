import tensorflow as tf
import os
from PIL import Image
import numpy as np

from char_level_encoder import CharLevelEncoder


def preprocess_image(image_path, target_size):
    # Load the image and preprocess it
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img = np.array(img).reshape(target_size + (1,)) / 255.0  # Normalize
    return img


def preprocess_content(txt_path, encoder):
    with open(txt_path, 'r') as file:
        content = file.read().strip()
    encoded_content = encoder.encode(content)
    return encoded_content


def load_data(image_dir, content_dir, target_size, encoder):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        txt_path = os.path.join(content_dir, img_file.replace('.png', '.txt'))

        if os.path.exists(txt_path):
            image = preprocess_image(img_path, target_size)
            encoded_content = preprocess_content(txt_path, encoder)
            yield image, encoded_content


def create_dataset(image_dir, content_dir, target_size, batch_size=32, shuffle=True, max_seq_len=512, num_chars=128):
    # Create the generator function

    encoder = CharLevelEncoder(max_sequence_length=max_seq_len, num_chars=num_chars)

    dataset = tf.data.Dataset.from_generator(
        lambda: load_data(image_dir, content_dir, target_size, encoder),
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=((max_seq_len, num_chars)), dtype=tf.float32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=250)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
