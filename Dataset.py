import numpy as np
import qrcode
import string
import random
import tensorflow as tf
from PIL import Image
from char_level_encoder import CharLevelEncoder  # Ensure this is correctly imported


# Functions for generating QR codes
def create_qr_code(content, error_correction=qrcode.constants.ERROR_CORRECT_L, resolution=512, max_shift=30,
                   max_rotation=60, noise_range=(1, 75)):
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=10,
        border=4,
    )

    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    img = apply_random_image_shift(img, max_shift)
    img = apply_random_image_rotation(img, max_rotation)

    if noise_range[1] > 0:
        img = add_random_image_noise(img, noise_range)

    img = img.resize((resolution, resolution))
    return img


def add_random_image_noise(image, noise_range=(1, 255)):
    noise_factor = random.randint(noise_range[0], noise_range[1])
    np_image = np.array(image)
    noise = np.random.randint(-noise_factor, noise_factor, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image.astype('uint8'))


def apply_random_image_shift(image, max_shift=10):
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    translation_matrix = (1, 0, shift_x, 0, 1, shift_y)
    background = Image.new('L', image.size, color=255)
    shifted_image = image.transform(image.size, Image.AFFINE, translation_matrix, fillcolor=255)
    background.paste(shifted_image)
    return background


def apply_random_image_rotation(image, max_angle=360):
    angle = random.uniform(-max_angle, max_angle)
    image = image.rotate(angle, expand=True, fillcolor=255)
    background = Image.new('L', image.size, color=255)
    background.paste(image)
    return background


def generate_random_data(population=None, max_length=300):
    if max_length == 0:
        return ''

    if population is None:
        population = string.ascii_uppercase + string.digits
    length = random.randint(1, max_length)
    content = ''.join(random.choices(population, k=length))
    return content


def generate_qr_code(repeats=3):
    """Generate a QR code and return its index for reference."""
    max_length = random.choice(
        [random.randint(0, 25), random.randint(26, 100), random.randint(101, 250), random.randint(251, 500)]
    )

    population = random.choice([string.ascii_uppercase, string.digits, string.ascii_uppercase + string.digits])

    content = generate_random_data(population, max_length)

    # Generate a clean QR code
    clean_qr_img = create_qr_code(content, noise_range=(0, 0), max_shift=0, max_rotation=0)

    # Generate noisy QR codes
    dirty_qr_imgs = [create_qr_code(content) for _ in range(repeats)]

    return content, clean_qr_img, dirty_qr_imgs


def normalize_image(image, target_size=(512, 512)):
    return np.array(image.convert('L')).reshape((target_size[0], target_size[1], 1)) / 255.0


def load_qr_code_data(target_size, encoder=None):
    """Infinite generator to create QR codes in memory without saving."""
    while True:
        content, clean, dirty = generate_qr_code()  # Generate QR codes in memory

        # Encode the content before yielding
        encoded_content = encoder.encode(content)

        # Yield clean QR code and encoded content
        yield normalize_image(clean, target_size), encoded_content

        for dirty_img in dirty:
            # Yield each dirty QR code and the same encoded content
            yield normalize_image(dirty_img, target_size), encoded_content


def create_dataset(target_size=(512, 512), batch_size=32, shuffle=False, max_seq_len=512, num_chars=128):
    encoder = CharLevelEncoder(max_sequence_length=max_seq_len, num_chars=num_chars)  # Initialize encoder
    dataset = tf.data.Dataset.from_generator(
        lambda: load_qr_code_data(target_size, encoder=encoder),  # Pass encoder
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(max_seq_len, num_chars), dtype=tf.float32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=250)

    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    for images, contents in dataset.take(1):  # Take one batch for example
        print("Images shape:", images.shape)
        print("Contents shape:", contents.shape)
        break

    return dataset


if __name__ == '__main__':
    target_size = (512, 512)  # Example target size for images
    dataset = create_dataset(target_size=target_size, batch_size=32)

    for images, contents in dataset.take(1):  # Take one batch for example
        print("Images shape:", images.shape)
        print("Contents shape:", contents.shape)
        break
