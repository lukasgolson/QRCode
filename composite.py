import os
import glob
import numpy as np
from PIL import Image


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_image(image_path):
    """Load an image and return it as a numpy array."""
    img = Image.open(image_path)
    return np.array(img)


def save_rgb_image(r, g=None, b=None, filename="image.png"):
    """Save an RGB image from the specified channels."""
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


def create_composites(input_dir, output_dir):
    """Combine layer channels into composite RGB images."""
    ensure_directory_exists(output_dir)

    # Get all layer files
    layer_files = glob.glob(os.path.join(input_dir, "layer_*_c*_output.png"))

    # Group images by layer
    layers = {}
    for file in layer_files:
        layer_number = int(file.split('layer_')[1].split('_')[0])
        if layer_number not in layers:
            layers[layer_number] = []
        layers[layer_number].append(file)

    # Create composite images
    for layer_number, files in layers.items():
        channel_images = [load_image(f) for f in files]
        num_channels = len(channel_images)

        # Combine channels into RGB images
        for i in range(0, num_channels, 3):
            r = channel_images[i]  # Red channel
            g = channel_images[i + 1] if (i + 1) < num_channels else None  # Green channel (if exists)
            b = channel_images[i + 2] if (i + 2) < num_channels else None  # Blue channel (if exists)

            # Create filename for the composite image
            sub_channel_idx = ""  # Sub-channel index (e.g., "1_2_3")

            if r is not None:
                sub_channel_idx += f"_{i + 1}"

            if g is not None:
                sub_channel_idx += f"_{i + 2}"

            if b is not None:
                sub_channel_idx += f"_{i + 3}"

            filename = f"{output_dir}/layer_{layer_number}_composite_sub_channel{sub_channel_idx}.png"

            save_rgb_image(r, g, b, filename)


if __name__ == "__main__":
    input_directory = "images"
    output_directory = "composites"
    create_composites(input_directory, output_directory)
