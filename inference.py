import os
import argparse
import pickle

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import glob
from datetime import datetime

from layers.involution import Involution
from model import SpatialTransformerInputHead


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


def load_latest_vectorizer(vectorizer_dir):
    vectorizer_files = glob.glob(os.path.join(vectorizer_dir, '*.pkl'))

    if not vectorizer_files:
        raise FileNotFoundError(f"No vectorizers found in directory: {vectorizer_dir}")

    latest_vectorizer_file = max(vectorizer_files, key=os.path.getmtime)
    print(f"Loading latest vectorizer: {latest_vectorizer_file}")

    from_disk = pickle.load(open(latest_vectorizer_file, "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])

    return new_v


# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((target_size, target_size))
    img_array = np.array(img).reshape((1, target_size, target_size, 1)) / 255.0  # Normalize
    return img_array


# Function to decode the predicted output to text
def decode_prediction(prediction, vectorizer):
    # Get the character index predictions (argmax across one-hot encoded characters)
    char_indices = np.argmax(prediction, axis=-1)[0]
    # Use the inverse lookup to map indices to characters
    decoded_text = ''.join([vectorizer.get_vocabulary()[i] for i in char_indices if i != 0])
    return decoded_text


# Main function to load image and run inference
def run_inference(image_path, model, vectorizer, target_size):
    # Preprocess the image
    image = preprocess_image(image_path, target_size)

    # Run the image through the model
    prediction = model.predict(image)

    # Decode the prediction to text
    decoded_text = decode_prediction(prediction, vectorizer)

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


    # Load the latest vectorizer
    try:
        vectorizer = load_latest_vectorizer(args.model_dir)
    except FileNotFoundError as e:
        print(e)
        exit(1)

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
        result = run_inference(args.image, model, vectorizer, 512)
        print(f"Predicted text content: {result}")
    else:
        print(f"Image not found at path: {args.image}")
