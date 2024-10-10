import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Reshape, TimeDistributed, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential

# Paths to directories
image_dir = 'data/images'
content_dir = 'data/contents'

# Helper function to load images
def load_image(filename, target_size=(256, 256)):
    img = Image.open(filename).convert('L')  # Convert image to grayscale
    img = img.resize(target_size)
    return np.array(img)

# Helper function to load text contents
def load_content(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
    return content

# Data generator for images and text content
class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, target_size=(256, 256), max_sequence_length=50, num_chars=1000):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.num_chars = num_chars
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.content_files = sorted([f for f in os.listdir(content_dir) if f.endswith('.txt')])

        # Load contents for fitting the vectorizer
        contents = self.load_contents()

        # Create and adapt the TextVectorization layer
        self.vectorizer = TextVectorization(output_mode='int',
                                            output_sequence_length=self.max_sequence_length,
                                            max_tokens=self.num_chars)
        self.vectorizer.adapt(contents)  # Fit the vectorizer on the contents

    def load_contents(self):
        contents = []
        for txt_file in self.content_files:
            with open(os.path.join(self.content_dir, txt_file), 'r') as file:
                contents.append(file.read().strip())
        return contents

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_x)
        return X, y

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

# Define model parameters
max_sequence_length = 32  # Based on dataset
num_chars = 1000  # Unique characters in the data

# Build the model
model = Sequential()

# Input layer for images
model.add(Input(shape=(256, 256, 1)))

# CNN for image feature extraction
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Second block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Third block
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Fourth block
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Flatten the output
model.add(Flatten())

# Dense layer to produce a feature vector for sequence prediction
model.add(Dense(512, activation='relu'))

# Reshape the output to match the number of time steps (max_sequence_length)
model.add(Reshape((max_sequence_length, -1)))  # Reshape to (sequence_length, feature_size)

# TimeDistributed Dense layers for outputting character predictions at each time step
model.add(TimeDistributed(Dense(num_chars, activation='softmax')))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Create the QRDataGenerator instance
batch_size = 32
epochs = 10
qr_data_gen = QRDataGenerator(image_dir, content_dir, batch_size=batch_size, max_sequence_length=max_sequence_length, num_chars=num_chars)

# Train the model
history = model.fit(qr_data_gen, epochs=epochs, steps_per_epoch=len(qr_data_gen))