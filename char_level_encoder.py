import pickle
import string

import numpy as np


class CharLevelEncoder:
    def __init__(self, max_sequence_length=512, custom_vocab=None):
        """
        Initialize the CharLevelEncoder with an optional custom vocabulary.

        :param max_sequence_length: Maximum length of sequences to be encoded.
        :param custom_vocab: Custom vocabulary as a string of characters (optional).
        """
        self.max_sequence_length = max_sequence_length
        self.vocab = custom_vocab if custom_vocab else self._create_default_vocab()
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.num_chars = len(self.vocab)

    @staticmethod
    def _create_default_vocab():
        """
        Create a default vocabulary of printable ASCII characters.

        :return: String of printable ASCII characters.
        """
        return string.ascii_letters + string.digits + string.punctuation + ' '

    def encode(self, texts):
        """
        Encode input texts into one-hot encoded sequences.

        :param texts: List of input strings.
        :return: Encoded one-hot sequences (numpy arrays).
        """
        encoded_texts = np.zeros((len(texts), self.max_sequence_length, self.num_chars), dtype=np.float32)

        for i, text in enumerate(texts):
            for j, char in enumerate(text[:self.max_sequence_length]):
                if char in self.char_to_index:
                    encoded_texts[i, j, self.char_to_index[char]] = 1.0

        return encoded_texts

    def decode(self, predictions):
        """
        Decode one-hot encoded model outputs back to text.

        :param predictions: Model output in one-hot encoded form (or integer sequences).
        :return: Decoded strings.
        """
        decoded_texts = []

        for prediction in predictions:
            decoded_text = ''
            for char_vector in prediction:
                char_index = np.argmax(char_vector)
                if char_index != 0:  # Assuming index 0 is padding or unknown
                    decoded_text += self.index_to_char[char_index]
            decoded_texts.append(decoded_text)

        return decoded_texts

    def encode_as_integers(self, texts):
        """
        Encode input texts into integer sequences (rather than one-hot).

        :param texts: List of input strings.
        :return: Integer encoded sequences (padded with zeros).
        """
        encoded_texts = np.zeros((len(texts), self.max_sequence_length), dtype=np.int32)

        for i, text in enumerate(texts):
            for j, char in enumerate(text[:self.max_sequence_length]):
                encoded_texts[i, j] = self.char_to_index.get(char, 0)  # Default to 0 if char not in vocab

        return encoded_texts

    def decode_from_integers(self, integer_sequences):
        """
        Decode integer sequences back into text.

        :param integer_sequences: Numpy array of integer sequences.
        :return: Decoded strings.
        """
        decoded_texts = []

        for sequence in integer_sequences:
            decoded_text = ''.join([self.index_to_char[idx] for idx in sequence if idx != 0])
            decoded_texts.append(decoded_text)

        return decoded_texts

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

# test code
