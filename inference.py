import argparse
import glob
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model
from char_level_encoder import CharLevelEncoder
from concurrent.futures import ThreadPoolExecutor


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


# Ensure "images" directory exists
def ensure_image_directory():
    if not os.path.exists('images'):
        os.makedirs('images')


# Function to save a channel or 1D output as a PNG
def save_channel_image(output_image, filename):
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0
    output_image = output_image.astype(np.uint8)
    img = Image.fromarray(output_image)
    img.save(filename)
    print(f"Saved: {filename}")


def save_rgb_image(r, g=None, b=None, filename="image.png"):
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
    print(f"Saved: {filename}")


# Function to handle image saving in parallel
def save_images_parallel(layer_number, output, prefix="layer"):
    ensure_image_directory()

    tasks = []
    with ThreadPoolExecutor() as executor:

        if len(output.shape) == 5:
            batch_size, dim1, dim2, dim3, channels = output.shape
            print(f"Saving {channels} channels from layer {layer_number} with dimensions {dim1}x{dim2}x{dim3}...")

            for i in range(0, channels, 3):
                # Extract RGB channels
                r = output[0, :, :, :, i]  # Red channel
                g = output[0, :, :, :, i + 1] if (i + 1) < channels else None  # Green channel (if exists)
                b = output[0, :, :, :, i + 2] if (i + 2) < channels else None  # Blue channel (if exists)

                sub_channel_idx = f"{i + 1}_{i + 2}_{i + 3}"  # Sub-channel index (e.g., "1_2_3")
                filename = f"images/{prefix}_{layer_number}_sub_channel_{sub_channel_idx}_output.png"

                tasks.append(executor.submit(save_rgb_image, r, g, b, filename))


        elif len(output.shape) == 4:  # (batch_size, width, height, channels)
            width, height, channels = output.shape[1:4]
            for ch in range(channels):
                output_image = output[0, :, :, ch]
                filename = f"images/{prefix}_{layer_number}_c{ch + 1}_output.png"
                tasks.append(executor.submit(save_channel_image, output_image, filename))

        elif len(output.shape) == 3:  # (batch_size, width, height)
            output_image = output[0]
            filename = f"images/{prefix}_{layer_number}_output.png"
            tasks.append(executor.submit(save_channel_image, output_image, filename))

        elif len(output.shape) == 2:  # (batch_size, features)
            output_image = output[0]  # .reshape(1, -1)  # Reshape to (1, n) for horizontal image
            filename = f"images/{prefix}_{layer_number}_output.png"
            tasks.append(executor.submit(save_channel_image, output_image, filename))

        elif len(output.shape) > 5:
            batch_size = output.shape[0]
            # Collapse all dimensions after the 3rd one into a single dimension
            dim1 = output.shape[1]
            dim2 = output.shape[2]
            dim3 = output.shape[3]
            combined_dims = np.prod(output.shape[4:])  # Product of all dimensions from the 4th onward
            flattened_output = output.reshape(batch_size, dim1, dim2, dim3, combined_dims)

            # Now, save the flattened output similar to the 5D case
            for i in range(0, combined_dims, 3):
                r = flattened_output[0, :, :, :, i]  # Red channel
                g = flattened_output[0, :, :, :, i + 1] if (i + 1) < combined_dims else None  # Green channel (if exists)
                b = flattened_output[0, :, :, :, i + 2] if (i + 2) < combined_dims else None  # Blue channel (if exists)

                sub_channel_idx = f"{i + 1}_{i + 2}_{i + 3}"  # Sub-channel index (e.g., "1_2_3")
                filename = f"images/{prefix}_{layer_number}_flattened_sub_channel_{sub_channel_idx}_output.png"
                tasks.append(executor.submit(save_rgb_image, r, g, b, filename))



        # Wait for all tasks to finish
        [task.result() for task in tasks]


# Function to extract activations for all layers at once
def get_all_layer_outputs(model, image):
    layer_outputs = [layer.output for layer in model.layers]  # Get output for each layer
    multi_output_model = Model(inputs=model.input, outputs=layer_outputs)  # Create a new model with multiple outputs
    return multi_output_model.predict(image)


# Function to print the model's architecture with layer IDs
def print_model_layers(model):
    print(f"{'Layer ID':<10} {'Layer Name':<25} {'Layer Type':<20} {'Output Shape':<20}")
    print("-" * 75)
    for idx, layer in enumerate(model.layers):
        print(f"{idx:<10} {layer.name:<25} {layer.__class__.__name__:<20} {layer.output_shape}")


# Main function to load image, run inference and save intermediate outputs
def run_inference(image_path, model, encoder, target_size, layer_number=None, save_output=False,
                  output_all_layers=False):
    image = preprocess_image(image_path, target_size)

    if output_all_layers:
        all_outputs = get_all_layer_outputs(model, image)  # Get all layer outputs in one pass
        for layer_number, layer_output in enumerate(all_outputs):
            print(f"Saving outputs for layer {layer_number}...")
            save_images_parallel(layer_number, layer_output)
    elif layer_number is not None:
        intermediate_output = get_all_layer_outputs(model, image)[layer_number]
        print(f"Intermediate output at layer {layer_number}:")
        if save_output:
            save_images_parallel(layer_number, intermediate_output)
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

        # if args.layer is None and not args.output_all_layers:
        print(f"Predicted text content: {result}")
    else:
        if not args.print_model:
            print("No image path provided for inference and --print_model not used. Exiting.")
