import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.utils import to_categorical, pad_sequences
from tqdm import tqdm

from char_level_encoder import CharLevelEncoder


class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, num_chars=128, target_size=(256, 256),
                 max_sequence_length=512, shuffle=False, **kwargs):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.num_chars = num_chars
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        self.valid_image_files = self.load_valid_files()
        self.encoder = CharLevelEncoder(max_sequence_length=self.max_sequence_length, num_chars=self.num_chars)
        self.current_index = 0

        # Shuffle the files initially if required
        if self.shuffle:
            np.random.shuffle(self.valid_image_files)

        super().__init__(**kwargs)

    def load_valid_files(self):
        # Create a set of valid text file names for quick lookup
        content_files = set(os.listdir(self.content_dir))

        # Use a list comprehension to collect valid image files
        valid_files = [
            img_file for img_file in tqdm(os.listdir(self.image_dir), desc="Finding valid QR pairs")
            if img_file.endswith('.png') and img_file.replace('.png', '.txt') in content_files
        ]

        print(f"Found {len(valid_files)} valid files.")
        return sorted(valid_files)

    def __len__(self):
        return int(np.floor(len(self.valid_image_files) / self.batch_size))

    def __getitem__(self, index):
        # Calculate the start index for the batch
        start_index = index * self.batch_size

        # Get the current batch of valid files
        batch_x = self.valid_image_files[start_index:start_index + self.batch_size]

        # If the current batch is empty (e.g., when index exceeds the number of batches),
        # we need to wrap around to ensure we always return a batch
        if len(batch_x) < self.batch_size:
            # If the end of valid files is reached, wrap around to the beginning
            # Fill the remaining items from the start of the valid files
            remaining_count = self.batch_size - len(batch_x)
            batch_x += self.valid_image_files[:remaining_count]

        # Ensure that we always return a batch of the specified batch_size
        assert len(batch_x) == self.batch_size, "Batch size does not match expected size."

        # Generate data for the current batch
        X, y = self.__data_generation(batch_x)

        return X, y

    def on_epoch_end(self):
        # Shuffle the valid files at the end of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.valid_image_files)
        self.current_index = 0  # Reset the current index

    def __data_generation(self, batch_x):
        X = []
        y = []

        for img_file in batch_x:
            img_path = os.path.join(self.image_dir, img_file)
            img = Image.open(img_path).convert('L')
            img = img.resize(self.target_size)
            X.append(np.array(img).reshape(self.target_size + (1,)) / 255.0)

            txt_file = img_file.replace('.png', '.txt')
            content = self.load_content(txt_file)

            one_hot_encoded = self.encoder.encode(content)

            y.append(one_hot_encoded)

        X = np.array(X)
        y = np.array(y)


        return X, y

    def load_content(self, filename):
        with open(os.path.join(self.content_dir, filename), 'r') as file:
            return file.read().strip()
