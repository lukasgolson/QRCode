import os

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.utils import to_categorical

from char_level_encoder import CharLevelEncoder


class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, num_chars=128, target_size=(256, 256),
                 max_sequence_length=512, shuffle=True, **kwargs):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.num_chars = num_chars
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        self.valid_image_files = self.load_valid_files()  # Load valid image-text pairs
        self.on_epoch_end()  # Shuffle at the start

        # Load contents for fitting the vectorizer
        self.contents = self.load_contents()

        # Create and adapt the TextVectorization layer
        self.encoder = CharLevelEncoder(max_sequence_length=self.max_sequence_length)

        self.current_index = 0  # Track the current index

        super().__init__(**kwargs)

    def load_valid_files(self):
        valid_files = []
        for img_file in os.listdir(self.image_dir):
            if img_file.endswith('.png'):
                txt_file = img_file.replace('.png', '.txt')
                if txt_file in os.listdir(self.content_dir):
                    valid_files.append(img_file)
        return sorted(valid_files)

    def load_contents(self):
        contents = []
        for txt_file in os.listdir(self.content_dir):
            if txt_file.endswith('.txt'):
                with open(os.path.join(self.content_dir, txt_file), 'r') as file:
                    contents.append(file.read().strip())
        return contents

    def __len__(self):
        return int(np.ceil(len(self.valid_image_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_x = self.valid_image_files[self.current_index:self.current_index + self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_x)

        # Update the current index for the next batch
        self.current_index += self.batch_size

        # Check if we've reached the end and need to reshuffle
        if self.current_index >= len(self.valid_image_files):
            self.current_index = 0  # Reset the index
            if self.shuffle:
                np.random.shuffle(self.valid_image_files)  # Reshuffle the valid files

        return X, y

    def on_epoch_end(self):
        # Shuffle the valid files and reset the current index
        if self.shuffle:
            np.random.shuffle(self.valid_image_files)
        self.current_index = 0  # Reset the current index at the end of the epoch

    def __data_generation(self, batch_x):
        X = []
        y = []

        for img_file in batch_x:
            # Load and preprocess image
            img_path = os.path.join(self.image_dir, img_file)
            img = Image.open(img_path).convert('L')
            img = img.resize(self.target_size)
            X.append(np.array(img).reshape(self.target_size + (1,)) / 255.0)

            # Load and vectorize corresponding text content
            txt_file = img_file.replace('.png', '.txt')
            content = self.load_content(txt_file)
            encoded_content = self.encoder.encode_as_integers([content])
            one_hot_encoded = to_categorical(encoded_content, num_classes=self.num_chars)
            y.append(one_hot_encoded)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        y = y.reshape(X.shape[0], 512, 128)  # Adjust based on your specific needs

        # print("Shape of X:", X.shape)  # Should be (batch_size, height, width, channels)
        # print("Shape of y:", y.shape)   # Should be (batch_size, max_sequence_length, num_chars)

        return X, y

    def load_content(self, filename):
        with open(os.path.join(self.content_dir, filename), 'r') as file:
            return file.read().strip()
