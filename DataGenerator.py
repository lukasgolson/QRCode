import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.utils import to_categorical
from tqdm import tqdm

from char_level_encoder import CharLevelEncoder


class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, num_chars=128, target_size=(512, 512),
                 max_sequence_length=512, shuffle=True, fraction_of_data=0.5, **kwargs):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.num_chars = num_chars
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle

        self.batch_division = 1 / fraction_of_data

        self.valid_image_files = self._load_valid_files()
        self.encoder = CharLevelEncoder(max_sequence_length=self.max_sequence_length, num_chars=self.num_chars)

        if self.shuffle:
            np.random.shuffle(self.valid_image_files)

        super().__init__(**kwargs)

    def _load_valid_files(self):
        content_files = set(os.listdir(self.content_dir))
        return sorted([
            img_file for img_file in os.listdir(self.image_dir)
            if img_file.endswith('.png') and img_file.replace('.png', '.txt') in content_files
        ])

    def __len__(self):
        actual_batches = len(self.valid_image_files) // self.batch_size
        return int(actual_batches // self.batch_division)

    def __getitem__(self, index):
        # Get the batch of files
        batch_files = self.valid_image_files[index * self.batch_size:(index + 1) * self.batch_size]

        # If the batch is smaller than expected, wrap around
        if len(batch_files) < self.batch_size:
            batch_files += self.valid_image_files[:self.batch_size - len(batch_files)]

        X, y = self._generate_data(batch_files)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_image_files)

    def _generate_data(self, batch_files):
        X, y = [], []

        for img_file in batch_files:
            img = Image.open(os.path.join(self.image_dir, img_file)).convert('L')
            img = img.resize(self.target_size)
            X.append(np.array(img).reshape(self.target_size + (1,)) / 255.0)

            content = self._load_content(img_file.replace('.png', '.txt'))
            y.append(self.encoder.encode(content))

        return np.array(X), np.array(y)

    def _load_content(self, txt_file):
        with open(os.path.join(self.content_dir, txt_file), 'r') as file:
            return file.read().strip()
