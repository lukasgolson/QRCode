import datetime
import os
import pickle

import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers, Layer, Model
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, LayerNormalization, Conv2DTranspose, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import TextVectorization, Dropout, Dense
from tensorflow.keras.utils import to_categorical

from layers.involution import Involution

# Paths to directories
image_dir = 'data/images'
content_dir = 'data/contents'

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def char_split(input_string):
    return tf.strings.unicode_split(input_string, 'UTF-8')


# Data generator for images and text content
class QRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, content_dir, batch_size=32, target_size=(256, 256),
                 max_sequence_length=512, num_chars=1000, shuffle=True, **kwargs):
        self.image_dir = image_dir
        self.content_dir = content_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_sequence_length = max_sequence_length
        self.num_chars = num_chars
        self.shuffle = shuffle
        self.valid_image_files = self.load_valid_files()  # Load valid image-text pairs
        self.on_epoch_end()  # Shuffle at the start

        # Load contents for fitting the vectorizer
        self.contents = self.load_contents()

        # Create and adapt the TextVectorization layer
        self.vectorizer = TextVectorization(
            split=char_split,
            output_mode='int',
            output_sequence_length=self.max_sequence_length,
            max_tokens=self.num_chars)
        self.vectorizer.adapt(self.contents)

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
            encoded_content = self.vectorizer([content]).numpy()[0]
            one_hot_encoded = to_categorical(encoded_content, num_classes=self.num_chars)
            y.append(one_hot_encoded)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Pad arrays if necessary to match batch size
        if len(y) < self.batch_size:
            X = np.pad(X, ((0, self.batch_size - len(X)), (0, 0), (0, 0), (0, 0)), mode='constant')
            y = np.pad(y, ((0, self.batch_size - len(y)), (0, 0), (0, 0)), mode='constant')

        return X, y

    def load_content(self, filename):
        with open(os.path.join(self.content_dir, filename), 'r') as file:
            return file.read().strip()


# Spatial Transformer Layer

class SpatialTransformerInputHead(Layer):
    def __init__(self, **kwargs):
        super(SpatialTransformerInputHead, self).__init__(**kwargs)

        # Localization head layers
        self.conv1 = Conv2D(14, (5, 5), padding='valid', activation='relu')
        self.pool1 = MaxPooling2D((2, 2), strides=2)
        self.conv2 = Conv2D(32, (5, 5), padding='valid', activation='relu')
        self.pool2 = MaxPooling2D((2, 2), strides=2)
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(84, activation='relu')
        self.output_layer = Dense(6, activation='linear',
                                  kernel_initializer='zeros',
                                  bias_initializer=lambda shape, dtype: tf.constant([1, 0, 0, 0, 1, 0], dtype=dtype))

    def generate_normalized_homo_meshgrids(self, inputs):
        batch_size = tf.shape(inputs)[0]
        _, H, W, _ = inputs.shape
        x_range = tf.range(W)
        y_range = tf.range(H)
        x_mesh, y_mesh = tf.meshgrid(x_range, y_range)
        x_mesh = (x_mesh / W - 0.5) * 2
        y_mesh = (y_mesh / H - 0.5) * 2
        y_mesh = tf.reshape(y_mesh, (*y_mesh.shape, 1))
        x_mesh = tf.reshape(x_mesh, (*x_mesh.shape, 1))
        ones_mesh = tf.ones_like(x_mesh)
        homogeneous_grid = tf.concat([x_mesh, y_mesh, ones_mesh], -1)
        homogeneous_grid = tf.reshape(homogeneous_grid, (-1, 3, 1))
        homogeneous_grid = tf.cast(homogeneous_grid, tf.float32)
        homogeneous_grid = tf.expand_dims(homogeneous_grid, 0)
        return tf.tile(homogeneous_grid, [batch_size, 1, 1, 1])

    def transform_grids(self, transformations, grids, inputs):
        trans_matrices = tf.reshape(transformations, (-1, 2, 3))
        batch_size = tf.shape(trans_matrices)[0]
        gs = tf.squeeze(grids, -1)
        reprojected_grids = tf.matmul(trans_matrices, gs, transpose_b=True)
        reprojected_grids = (tf.linalg.matrix_transpose(reprojected_grids) + 1) * 0.5
        _, H, W, _ = inputs.shape
        return tf.multiply(reprojected_grids, [W, H])

    def generate_four_neighbors_from_reprojection(self, inputs, reprojected_grids):
        _, H, W, _ = inputs.shape
        x, y = tf.split(reprojected_grids, 2, axis=-1)
        x1 = tf.floor(x)
        x1 = tf.cast(x1, tf.int32)
        x2 = x1 + tf.constant(1)
        y1 = tf.floor(y)
        y1 = tf.cast(y1, tf.int32)
        y2 = y1 + tf.constant(1)
        y_max = tf.constant(H - 1, dtype=tf.int32)
        x_max = tf.constant(W - 1, dtype=tf.int32)
        zero = tf.zeros([1], dtype=tf.int32)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)
        x2_safe = tf.clip_by_value(x2, zero, x_max)
        y2_safe = tf.clip_by_value(y2, zero, y_max)
        return x1_safe, y1_safe, x2_safe, y2_safe

    def bilinear_sample(self, inputs, reprojected_grids):
        x1, y1, x2, y2 = self.generate_four_neighbors_from_reprojection(inputs, reprojected_grids)
        x1y1 = tf.concat([y1, x1], -1)
        x1y2 = tf.concat([y2, x1], -1)
        x2y1 = tf.concat([y1, x2], -1)
        x2y2 = tf.concat([y2, x2], -1)
        pixel_x1y1 = tf.gather_nd(inputs, x1y1, batch_dims=1)
        pixel_x1y2 = tf.gather_nd(inputs, x1y2, batch_dims=1)
        pixel_x2y1 = tf.gather_nd(inputs, x2y1, batch_dims=1)
        pixel_x2y2 = tf.gather_nd(inputs, x2y2, batch_dims=1)
        x, y = tf.split(reprojected_grids, 2, axis=-1)
        wx = tf.concat([tf.cast(x2, tf.float32) - x, x - tf.cast(x1, tf.float32)], -1)
        wx = tf.expand_dims(wx, -2)
        wy = tf.concat([tf.cast(y2, tf.float32) - y, y - tf.cast(y1, tf.float32)], -1)
        wy = tf.expand_dims(wy, -1)
        Q = tf.concat([pixel_x1y1, pixel_x1y2, pixel_x2y1, pixel_x2y2], -1)
        Q_shape = tf.shape(Q)
        Q = tf.reshape(Q, (Q_shape[0], Q_shape[1], 2, 2))
        r = wx @ Q @ wy
        _, H, W, channels = inputs.shape
        r = tf.reshape(r, (-1, H, W, 1))
        return r

    def call(self, inputs):
        # Compute transformations using localization head
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        transformations = self.output_layer(x)

        # Apply spatial transformation
        grids = self.generate_normalized_homo_meshgrids(inputs)
        reprojected_grids = self.transform_grids(transformations, grids, inputs)
        return self.bilinear_sample(inputs, reprojected_grids)


gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")


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


def create_involution_architecture(input_tensor, length, channels=16, group_number=1, downscale_factor=2):
    x = input_tensor

    for i in range(length):
        global involution_id
        involution_id += 1

        channels_count = channels * (2 ** i)

        x = keras.layers.Conv2D(channels_count, (1, 1), activation='relu')(x)

        x, _ = Involution(
            channel=channels_count, group_number=group_number, kernel_size=3, stride=2, reduction_ratio=2,
            name=f'involution_{involution_id}'
        )(x)

        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

    return x


def create_model(input_shape, max_sequence_length, num_chars):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Instantiate the SpatialTransformerInputHead
    processing_head = SpatialTransformerInputHead()(inputs)  # Ensure the output is used correctly

    print(processing_head.shape)

    # Build the involution architecture

    x = processing_head

    # upscale by 2
    x = create_involution_architecture(x, 4, 16, 4)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization
    x = layers.Dense(512, activation='relu')(x)

    x = Dropout(0.25)(x)

    # reduce to 512X512X1
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    # Flatten and reshape for sequence prediction
    sequence = layers.Flatten()(x)
    sequence = layers.Dense(max_sequence_length * num_chars, activation='relu')(sequence)
    sequence = layers.Reshape((max_sequence_length, -1))(sequence)  # Reshape to (sequence_length, feature_size)

    pos_encoding = positional_encoding(max_sequence_length, x.shape[-1])
    sequence += pos_encoding

    sequence = layers.Dense(512, activation='relu')(sequence)
    sequence = layers.BatchNormalization()(sequence)

    outputs = layers.TimeDistributed(layers.Dense(num_chars, activation='softmax'))(sequence)

    return Model(inputs, outputs, name='qr_model')


def get_compiled_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    input_shape = (target_image_size, target_image_size, 1)  # Define the input shape for the images

    model = create_model(input_shape, max_sequence_length, num_chars)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def make_or_restore_model(max_sequence_length=512, num_chars=128, target_image_size=512):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(max_sequence_length, num_chars, target_image_size)


def run_training(epochs=1, batch_size=16):
    max_sequence_length = 512
    num_chars = 128
    target_image_size = 512

    strategy = tf.distribute.MirroredStrategy()

    tf.keras.backend.clear_session()

    with strategy.scope():
        model = make_or_restore_model(max_sequence_length, num_chars, target_image_size)
        model.summary()

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [

        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}.keras", save_freq="epoch"
        )

    ]

    qr_data_gen = QRDataGenerator(image_dir, content_dir, batch_size=batch_size,
                                  max_sequence_length=max_sequence_length,
                                  num_chars=num_chars, target_size=(target_image_size, target_image_size))

    model.fit(qr_data_gen, epochs=epochs, steps_per_epoch=len(qr_data_gen), callbacks=callbacks)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the model
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
    model.save(os.path.join(save_path, f'qr_model_{date}.keras'))

    vectorizer_config = qr_data_gen.vectorizer.get_config()
    vectorizer_weights = qr_data_gen.vectorizer.get_weights()

    with open(os.path.join(save_path, f'vectorizer_{date}.pckl'), "wb") as f:
        pickle.dump({'config': vectorizer_config, 'weights': vectorizer_weights}, f)


if __name__ == "__main__":
    run_training(epochs=16, batch_size=8)
