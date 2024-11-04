import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model

from char_level_encoder import CharLevelEncoder
from train import masked_categorical_crossentropy

import ImageCleanModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs (including info, warnings, errors)


# Function to load the latest model from the model directory
def load_latest_model(model_dir):
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))
    print(f"Found model files: {model_files}")  # Debug statement

    if not model_files:
        raise FileNotFoundError(f"No models found in directory: {model_dir}")

    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"Loading latest model: {latest_model_file}")
    model = load_model(latest_model_file, custom_objects={'loss_func': ImageCleanModel.loss_func}, compile=False)

    model.compile(optimizer='adamw', loss=ImageCleanModel.loss_func, metrics=['accuracy'])

    print("Model loaded successfully.")  # Debug statement

    model.summary()

    return model


# Function to preprocess the image
def preprocess_image(image_path, target_size):
    print(f"Preprocessing image: {image_path} with target size: {target_size}")  # Debug statement
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((target_size, target_size))
    img_array = np.array(img).reshape((1, target_size, target_size, 1)) / 255.0  # Normalize
    print(f"Image preprocessed with shape: {img_array.shape}")  # Debug statement
    return img_array


# Ensure "images" directory exists
def ensure_image_directory():
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Created 'images' directory.")  # Debug statement
    else:
        print("'images' directory already exists.")  # Debug statement


# Function to save a channel or 1D output as a PNG
def save_channel_image(output_image, filename):
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0
    output_image = output_image.astype(np.uint8)

    output_image = np.squeeze(output_image)


    img = Image.fromarray(output_image)
    img.save(filename)
    print(f"Saved channel image: {filename}")  # Debug statement


def save_rgb_image(r, g=None, b=None, filename="image.png"):
    print(f"Saving RGB image: {filename}")  # Debug statement
    if g is None:
        g = r
    if b is None:
        b = r

    def normalize_channel(channel):
        return (255.0 * (channel - np.min(channel)) / (np.max(channel) - np.min(channel))).astype(np.uint8)

    r = normalize_channel(r)
    g = normalize_channel(g)
    b = normalize_channel(b)

    rgb_image = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb_image)
    img.save(filename)
    print(f"Saved RGB image: {filename}")  # Debug statement


# Function to handle image saving in parallel
def save_images_parallel(layer_number, output, prefix="layer"):
    ensure_image_directory()
    print(f"Saving images in parallel for layer {layer_number}.")  # Debug statement

    tasks = []
    with ThreadPoolExecutor() as executor:
        batch_size = output.shape[0]
        print(f"Output shape: {output.shape}, Batch size: {batch_size}")  # Debug statement

        # Handle different dimensional cases (add debug statements as needed)
        try:
            if len(output.shape) == 2:  # (batch_size, features)
                output_image = output[0].reshape(1, -1)  # Reshape to horizontal image
                filename = f"images/{prefix}_{layer_number}_output.png"
                tasks.append(executor.submit(save_channel_image, output_image, filename))

            elif len(output.shape) == 3:  # (batch_size, width, height)
                output_image = output[0]
                filename = f"images/{prefix}_{layer_number}_output.png"
                tasks.append(executor.submit(save_channel_image, output_image, filename))

            elif len(output.shape) == 4:  # (batch_size, width, height, channels)
                for ch in range(output.shape[-1]):
                    output_image = output[0, :, :, ch]
                    filename = f"images/{prefix}_{layer_number}_c{ch + 1}_output.png"
                    tasks.append(executor.submit(save_channel_image, output_image, filename))

            elif len(output.shape) == 5:  # (batch_size, dim1, dim2, dim3, channels)
                for i in range(0, output.shape[-1], 3):
                    r = output[0, :, :, :, i]
                    g = output[0, :, :, :, i + 1] if (i + 1) < output.shape[-1] else None
                    b = output[0, :, :, :, i + 2] if (i + 2) < output.shape[-1] else None
                    filename = f"images/{prefix}_{layer_number}_rgb_{i + 1}_{i + 2}_{i + 3}.png"
                    tasks.append(executor.submit(save_rgb_image, r, g, b, filename))

            # Wait for all tasks to finish
            [task.result() for task in tasks]
            print("All images saved successfully.")  # Debug statement

        except Exception as e:
            print(f"Error saving images for layer {layer_number}: {e}")


# Function to extract activations for all layers at once
def get_all_layer_outputs(model, image):
    print("Extracting outputs for all layers...")  # Debug statement
    start_time = time.time()  # Start timing
    layer_outputs = [layer.output for layer in model.layers if hasattr(layer, 'output')]
    print(f"Total layers: {len(layer_outputs)}")  # Debug statement
    multi_output_model = Model(inputs=model.input, outputs=layer_outputs)  # Create a new model with multiple outputs

    print("Model created with multiple outputs.")  # Debug statement

    multi_output_model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model compiled successfully.")  # Debug statement

    outputs = multi_output_model.predict(image)
    end_time = time.time()  # End timing
    print(f"Outputs extracted in {end_time - start_time:.2f} seconds.")  # Timing debug statement
    return outputs


# Function to print the model's architecture with layer IDs
def print_model_layers(model):
    print(f"{'Layer ID':<10} {'Layer Name':<25} {'Layer Type':<20} {'Output Shape':<20}")
    print("-" * 75)
    for idx, layer in enumerate(model.layers):
        output_shape = layer.output.shape if hasattr(layer, 'output') else "N/A"
        print(f"{idx:<10} {layer.name:<25} {layer.__class__.__name__:<20} {output_shape}")


# Main function to load image, run inference and save intermediate outputs
def run_inference(image_path, model, encoder, layer_number=None, save_output=False,
                  output_all_layers=False, output_is_image=False):
    print(f"Running inference on image: {image_path}")  # Debug statement

    # get model input shape

    input_shape = model.input_shape[1:]

    print(f"Model input shape: {input_shape}")  # Debug statement

    target_size = input_shape[0]  # Set target size to model input shape

    image = preprocess_image(image_path, target_size)

    # reshape image to match model input shape; including channel dimension if needed
    if len(input_shape) == 4:
        image = image.reshape(input_shape)

    if output_all_layers:
        all_outputs = get_all_layer_outputs(model, image)  # Get all layer outputs in one pass

        for layer_number, layer_output in enumerate(all_outputs):
            print(f"Layer {layer_number} output shape: {layer_output.shape}")  # Debug statement
            print(f"Saving outputs for layer {layer_number}...")  # Debug statement
            save_images_parallel(layer_number, layer_output)

    elif layer_number is not None:
        intermediate_output = get_all_layer_outputs(model, image)[layer_number]
        print(f"Intermediate output at layer {layer_number}:")
        if save_output:
            save_images_parallel(layer_number, intermediate_output)

    prediction = model.predict(image)

    if output_is_image:
        output_image = prediction[0]
        save_channel_image(output_image, 'output_image.png')
        print("Output image saved.")
        return None
    else:
        decoded_text = encoder.decode(prediction)
        print(f"Decoded text: {decoded_text}")  # Debug statement
        return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Code Inference Application")
    parser.add_argument('--image', type=str, help="Path to the input image")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory containing trained models")
    parser.add_argument('--layer', type=int, help="Layer number to examine intermediate output", default=None)
    parser.add_argument('--save_output', action='store_true', help="Save the intermediate layer output as PNG")
    parser.add_argument('--output_all_layers', action='store_true', help="Output all layers of the model as images")
    parser.add_argument('--print_model', action='store_true', help="Print the model architecture with layer IDs")

    parser.add_argument("--image_out", type=bool, default=False, help="Model output is an image")

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

        result = run_inference(args.image, model, encoder, args.layer, args.save_output, args.output_all_layers, output_is_image=args.image_out)

        if result is not None:
            print(f"Predicted text content: {result}")
    else:
        if not args.print_model:
            print("No image path provided for inference and --print_model not used. Exiting.")

    print("Inference complete.")
