import datetime
import os

import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras import layers
from tensorflow.keras.layers import TextVectorization, Dropout
from tensorflow.keras.utils import to_categorical

from layers.involution import Involution

# Paths to directories
image_dir = 'data/images'
content_dir = 'data/contents'


# Data generator for images and text content
class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, target_size=(256, 256), max_sequence_length=512,
                 num_chars=1000, shuffle=True, **kwargs):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.num_chars = num_chars
        self.shuffle = shuffle
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.content_files = sorted([f for f in os.listdir(content_dir) if f.endswith('.txt')])

        # Load contents for fitting the vectorizer
        self.contents = self.load_contents()  # Preload contents

        # Create and adapt the TextVectorization layer
        self.vectorizer = TextVectorization(output_mode='int',
                                            output_sequence_length=self.max_sequence_length,
                                            max_tokens=self.num_chars)
        self.vectorizer.adapt(self.contents)  # Fit the vectorizer on the contents

        # Shuffle indices if needed
        self.on_epoch_end()

        super().__init__(**kwargs)

    def load_contents(self):
        contents = []
        for txt_file in self.content_files:
            with open(os.path.join(self.content_dir, txt_file), 'r') as file:
                contents.append(file.read().strip())
        return contents

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of image files for this batch
        batch_x = [self.image_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_x)
        return X, y

    def on_epoch_end(self):
        # Update indexes after each epoch and shuffle if enabled
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_x):
        X = np.empty((len(batch_x), *self.target_size, 1))
        y = np.empty((len(batch_x), self.max_sequence_length, self.num_chars))

        for i, img_file in enumerate(batch_x):
            # Load and preprocess image
            img = Image.open(os.path.join(self.image_dir, img_file)).convert('L')
            img = img.resize(self.target_size)
            X[i,] = np.array(img).reshape(self.target_size + (1,)) / 255.0

            # Load and vectorize corresponding text content
            content = self.load_content(img_file.replace('.png', '.txt'))
            encoded_content = self.vectorizer([content]).numpy()[0]  # Use the TextVectorization layer to encode
            y[i,] = to_categorical(encoded_content, num_classes=self.num_chars)  # One-hot encode the output

        return X, y

    def load_content(self, filename):
        with open(os.path.join(self.content_dir, filename), 'r') as file:
            return file.read().strip()


# Spatial Transformer Layer


gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")

# Define the strategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

# Print the number of devices being used for training
print(f"Number of devices: {strategy.num_replicas_in_sync}")


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


involution_id = 0


def create_involution_architecture(input_tensor, length):
    head = input_tensor

    for i in range(length):
        global involution_id
        involution_id += 1

        y, _ = Involution(
            channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name=f'involution_{involution_id}'
        )(head)

        y = keras.layers.Conv2D(3, (1, 1), activation='relu')(y)

        head = y

    return head


def create_model(input_shape, max_sequence_length, num_chars):
    inputs = layers.Input(shape=input_shape)

    # dense layer for initial processing

    head = inputs

    # Build the involution architecture
    x = create_involution_architecture(head, 2)

    x = layers.BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.25)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)

    x = Dropout(0.25)(x)




# Flatten and reshape for sequence prediction
    sequence = layers.Flatten()(x)
    sequence = layers.Dense(max_sequence_length, activation='relu')(sequence)
    sequence = layers.Reshape((max_sequence_length, -1))(sequence)  # Reshape to (sequence_length, feature_size)

    # Add positional encoding to the sequence
    pos_encoding = positional_encoding(max_sequence_length, sequence.shape[-1])
    sequence += pos_encoding

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(sequence)

    return Model(inputs, outputs, name='qr_model')


# Define model parameters
max_sequence_length = 512  # Based on dataset
num_chars = 128  # Unique characters in the data
target_image_size = 512  # Image pixel size (length and width)

input_shape = (target_image_size, target_image_size, 1)  # Define the input shape for the images
model = create_model(input_shape, max_sequence_length, num_chars)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Create the QRDataGenerator instance
batch_size = 32
epochs = 1
qr_data_gen = QRDataGenerator(image_dir, content_dir, batch_size=batch_size, max_sequence_length=max_sequence_length,
                              num_chars=num_chars, target_size=(target_image_size, target_image_size))

# Train the model
history = model.fit(qr_data_gen, epochs=epochs, steps_per_epoch=len(qr_data_gen))

# Get current time and date
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the model
save_path = 'models'
os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
model.save(os.path.join(save_path, f'qr_model_{date}.h5'))
