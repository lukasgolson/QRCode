import os
import argparse
import pickle

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

import glob
from datetime import datetime

from char_level_encoder import CharLevelEncoder
from layers.SpatialTransformer import SpatialTransformerInputHead
from layers.involution import Involution


# Function to load the latest model from the model directory
def load_latest_model(model_dir):
    # Find all .keras files in the directory
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))

    if not model_files:
        raise FileNotFoundError(f"No models found in directory: {model_dir}")

    # Extract modification times and sort by latest
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"Loading latest model: {latest_model_file}")
    return load_model(latest_model_file, custom_objects={'SpatialTransformerInputHead': SpatialTransformerInputHead,
                                                         'Involution': Involution})



# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((target_size, target_size))
    img_array = np.array(img).reshape((1, target_size, target_size, 1)) / 255.0  # Normalize
    return img_array


# Function to decode the predicted output to text
def decode_prediction(prediction, encoder):
    # Assuming prediction shape is (1, max_sequence_length, num_chars)
    char_indices = np.argmax(prediction, axis=-1)  # Get indices of max predictions across the character dimension to undo one-hot encoding


    result = ''
    for char_index in char_indices[0]:
        result += encoder.index_to_char[char_index]


    return result  # Return the first element since we expect one result


# Main function to load image and run inference
def run_inference(image_path, model, encoder, target_size):
    # Preprocess the image
    image = preprocess_image(image_path, target_size)

    # Run the image through the model
    prediction = model.predict(image)

    print(prediction)


    # Decode the prediction to text
    decoded_text = decode_prediction(prediction, encoder)

    return decoded_text


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="QR Code Inference Application")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory containing trained models")

    args = parser.parse_args()

    # validate the image path
    if not os.path.exists(args.image):
        print(f"Image not found at path: {args.image}")
        exit(1)

    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        exit(1)


    # Load the char level encoder
    encoder = CharLevelEncoder()


    # Load the latest model
    try:
        model = load_latest_model(args.model_dir)
    except FileNotFoundError as e:
        print(e)
        exit(1)



    # Define parameters (must match what was used during training)
    max_sequence_length = 512
    num_chars = 128



    # Ensure the image path exists
    if os.path.exists(args.image):
        result = run_inference(args.image, model, encoder, 512)
        print(f"Predicted text content: {result}")
    else:
        print(f"Image not found at path: {args.image}")
