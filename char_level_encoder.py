import pickle
import string

import numpy as np


class CharLevelEncoder:
    def __init__(self, max_sequence_length=512, num_chars=128, custom_vocab=None):
        """
        Initialize the CharLevelEncoder with an optional custom vocabulary.

        :param max_sequence_length: Maximum length of sequences to be encoded.
        :param custom_vocab: Custom vocabulary as a string of characters (optional).
        """
        self.max_sequence_length = max_sequence_length
        self.vocab = custom_vocab if custom_vocab else self._create_default_vocab()
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.num_chars = num_chars

    @staticmethod
    def _create_default_vocab():
        """
        Create a default vocabulary of printable ASCII characters.

        :return: String of printable ASCII characters.
        """
        return string.printable + ' '

    def print_vocabulary(self):
        # Print the header
        print(f"{'Index':<6} {'Character':<10}")
        print("-" * 16)  # Separator line

        # Print each index and character
        for index, char in self.index_to_char.items():
            print(f"{index:<6} {char:<10}")

    def encode(self, text):
        """
        Encode input texts into one-hot encoded sequences.

        :param texts: List of input strings.
        :return: Encoded one-hot sequences (numpy arrays).
        """
        encoded_texts = np.zeros((self.max_sequence_length, self.num_chars), dtype=np.float32)

        for j, char in enumerate(text[:self.max_sequence_length]):
            if char in self.char_to_index:
                encoded_texts[j, self.char_to_index[char]] = 1.0

        return encoded_texts

    def decode(self, prediction: np.ndarray) -> str:
        """
        Decode one-hot encoded model outputs back to text.

        :param prediction: Model output in one-hot encoded form (or integer sequences).
        :return: Decoded string.
        """

        # Check if the prediction has the expected dimensions
        if prediction.ndim != 3:
            raise ValueError("Expected prediction to be a 3D array (batch_size, max_sequence_length, num_chars).")


        prediction = np.exp(prediction) / np.sum(np.exp(prediction), axis=-1, keepdims=True)

        predicted_classes = np.argmax(prediction, axis=-1)  # This will give us indices for each of the 512 steps





        decoded_text = []
        # Assuming we're only interested in the first sequence in the batch (index 0)
        for char_index in predicted_classes[0]:  # Select the first sequence in the batch
            if char_index < len(self.index_to_char):  # Ensure index is within vocabulary range
                decoded_text.append(self.index_to_char[char_index])  # Append corresponding character
            else:
                decoded_text.append('�')  # Append a placeholder for unknown characters

        return ''.join(decoded_text)  # Join list into a single string for efficiency

    def save_encoder(self, filepath):
        """
        Save the encoder (vocabulary) to a file.

        :param filepath: Path to save the encoder.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.vocab, f)

    def load_encoder(self, filepath):
        """
        Load the encoder (vocabulary) from a file.

        :param filepath: Path to load the encoder from.
        """
        with open(filepath, "rb") as f:
            self.vocab = pickle.load(f)
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def vocab_size(self):
        """
        Get the size of the vocabulary.

        :return: Number of unique characters in the vocabulary.
        """
        return len(self.vocab)
