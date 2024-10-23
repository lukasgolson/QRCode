import pickle
import string

import numpy as np


class CharLevelEncoder:
    def __init__(self, max_sequence_length=512, num_chars=129, custom_vocab=None):
        """
        Initialize the CharLevelEncoder with an optional custom vocabulary.

        :param max_sequence_length: Maximum length of sequences to be encoded.
        :param custom_vocab: Custom vocabulary as a string of characters (optional).
        :param eos_char: Character to use as the end-of-sequence marker.
        """
        self.max_sequence_length = max_sequence_length
        self.eos_char = '<EOS>'
        self.vocab = custom_vocab if custom_vocab else self._create_default_vocab(self.eos_char)
        self.char_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_char = {idx: word for idx, word in enumerate(self.vocab)}
        self.num_chars = num_chars


        self.print_vocabulary()

    @staticmethod
    def _create_default_vocab(eos_char):
        """
        Create a default vocabulary of printable ASCII characters, ensuring EOS character is included.

        :return: Two-way dictionaries mapping strings to indices and vice versa.
        """
        # List of printable characters and include space
        printable_chars = string.printable

        # split by character to get words

        printable_chars = list(printable_chars)


        # Add special tokens
        printable_chars.append(eos_char)  # Add EOS character
        printable_chars.append('<PAD>')    # Add padding character if needed

        return printable_chars

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

        :param text: Input string to be encoded.
        :return: Encoded one-hot sequences (numpy arrays).
        """
        encoded_texts = np.zeros((self.max_sequence_length, self.num_chars), dtype=np.float32)

        # Encode the input text characters
        for j, char in enumerate(text[:self.max_sequence_length]):
            if char in self.char_to_index:
                encoded_texts[j, self.char_to_index[char]] = 1.0

        # Add the EOS character at the end of the sequence
        if len(text) >= self.max_sequence_length:
            encoded_texts[-1, self.char_to_index[self.eos_char]] = 1.0
        else:
            encoded_texts[len(text), self.char_to_index[self.eos_char]] = 1.0

        return encoded_texts

    def decode(self, prediction: np.ndarray) -> str:
        """
        Decode one-hot encoded model outputs back to text.

        :param prediction: Model output in one-hot encoded form (or integer sequences).
        :return: Decoded string.
        """
        if prediction.ndim != 3:
            raise ValueError("Expected prediction to be a 3D array (batch_size, max_sequence_length, num_chars).")

        predicted_classes = np.argmax(prediction, axis=-1)

        decoded_text = []
        for char_index in predicted_classes[0]:
            if char_index == self.char_to_index[self.eos_char]:  # Check for EOS
                break
            decoded_text.append(self.index_to_char.get(char_index, '�'))  # Handle unknown characters

        return ''.join(decoded_text)

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
