import argparse
import glob
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model

from char_level_encoder import CharLevelEncoder


# Function to load the latest model from the model directory
def load_latest_model(model_dir):
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))

    if not model_files:
        raise FileNotFoundError(f"No models found in directory: {model_dir}")

    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"Loading latest model: {latest_model_file}")
    return load_model(latest_model_file)


# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((target_size, target_size))
    img_array = np.array(img).reshape((1, target_size, target_size, 1)) / 255.0  # Normalize
    return img_array


# Function to decode the predicted output to text
def decode_prediction(prediction, encoder):
    encoder.print_vocabulary()
    return encoder.decode(prediction)


# Ensure "images" directory exists
def ensure_image_directory():
    if not os.path.exists('images'):
        os.makedirs('images')


# Function to save each channel as a separate PNG
def save_intermediate_output(output, layer_number, prefix="layer"):
    ensure_image_directory()

    if len(output.shape) == 4:  # (batch_size, width, height, channels)
        width, height, channels = output.shape[1:4]
        for ch in range(channels):
            output_image = output[0, :, :, ch]
            output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0
            output_image = output_image.astype(np.uint8)

            filename = f"images/{prefix}_{layer_number}_c{ch+1}_output.png"
            img = Image.fromarray(output_image)
            img.save(filename)
            print(f"Saved: {filename}")

    elif len(output.shape) == 3:  # (batch_size, width, height)
        output_image = output[0]
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0
        output_image = output_image.astype(np.uint8)

        filename = f"images/{prefix}_{layer_number}_output.png"
        img = Image.fromarray(output_image)
        img.save(filename)
        print(f"Saved: {filename}")

    elif len(output.shape) == 2:  # (batch_size, features)
        output_image = output[0].reshape(1, -1)  # Reshape to (1, n) for horizontal image
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0
        output_image = output_image.astype(np.uint8)

        filename = f"images/{prefix}_{layer_number}_output.png"
        img = Image.fromarray(output_image)
        img.save(filename)
        print(f"Saved: {filename}")


# Function to get output of a specific layer in the model
def get_intermediate_output(model, image, layer_number):
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layer_number].output)
    intermediate_output = intermediate_model.predict(image)
    return intermediate_output


# Function to print the model's architecture with layer IDs
def print_model_layers(model):
    print(f"{'Layer ID':<10} {'Layer Name':<25} {'Layer Type':<20} {'Output Shape':<20}")
    print("-" * 75)
    for idx, layer in enumerate(model.layers):
        print(f"{idx:<10} {layer.name:<25} {layer.__class__.__name__:<20} {layer.output_shape}")


# Function to save outputs of all layers in the model
def save_all_layer_outputs(model, image):
    ensure_image_directory()

    for layer_number, layer in enumerate(model.layers):
        intermediate_output = get_intermediate_output(model, image, layer_number)
        print(f"Saving outputs for layer {layer_number} ({layer.name})...")
        save_intermediate_output(intermediate_output, layer_number)


# Main function to load image and run inference
def run_inference(image_path, model, encoder, target_size, layer_number=None, save_output=False, output_all_layers=False):
    image = preprocess_image(image_path, target_size)

    if output_all_layers:
        save_all_layer_outputs(model, image)
    elif layer_number is not None:
        intermediate_output = get_intermediate_output(model, image, layer_number)
        print(f"Intermediate output at layer {layer_number}:")
        print(intermediate_output)

        if save_output:
            save_intermediate_output(intermediate_output, layer_number)
    else:
        prediction = model.predict(image)
        print("Model prediction output:", prediction)
        decoded_text = decode_prediction(prediction, encoder)
        return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Code Inference Application")
    parser.add_argument('--image', type=str, help="Path to the input image")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory containing trained models")
    parser.add_argument('--layer', type=int, help="Layer number to examine intermediate output", default=None)
    parser.add_argument('--save_output', action='store_true', help="Save the intermediate layer output as PNG")
    parser.add_argument('--output_all_layers', action='store_true', help="Output all layers of the model as images")
    parser.add_argument('--print_model', action='store_true', help="Print the model architecture with layer IDs")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        exit(1)

    encoder = CharLevelEncoder()

    try:
        model = load_latest_model(args.model_dir)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Print the model architecture if requested
    if args.print_model:
        print_model_layers(model)

    # Ensure the image path exists for inference
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found at path: {args.image}")
            exit(1)

        result = run_inference(args.image, model, encoder, 512, args.layer, args.save_output, args.output_all_layers)

        if args.layer is None and not args.output_all_layers:
            print(f"Predicted text content: {result}")
    else:
        if not args.print_model:
            print("No image path provided for inference and --print_model not used. Exiting.")
