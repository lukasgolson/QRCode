import numpy as np
import qrcode
import string
import random
import tensorflow as tf
from PIL import Image
from char_level_encoder import CharLevelEncoder  # Ensure this is correctly imported


# Functions for generating QR codes
def create_qr_code(content, error_correction=qrcode.constants.ERROR_CORRECT_L):
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=10,
        border=4,
    )

    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    return img


def dirty_qr_code(img, max_shift=30, max_rotation=60, noise_range=(1, 75)):
    img = img.copy()
    img = apply_random_image_shift(img, max_shift)
    img = apply_random_image_rotation(img, max_rotation)

    if noise_range[1] > 0:
        img = add_random_image_noise(img, noise_range)

    return img


def add_random_image_noise(image, noise_range=(1, 255)):
    noise_factor = random.randint(noise_range[0], noise_range[1] + 1)
    np_image = np.array(image)

    noise = np.random.randint(-noise_factor, noise_factor + 1, np_image.shape)
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


def generate_qr_code(repeats=3, max_sequence_length=500):
    """Generate a QR code and return its index for reference."""
    content_length = random.randint(0, max_sequence_length)

    population = random.choice([string.ascii_uppercase, string.digits, string.ascii_uppercase + string.digits])

    content = generate_random_data(population, content_length)

    # Generate a clean QR code
    prototype = create_qr_code(content)

    # Generate noisy QR codes
    dirty_qr_imgs = [dirty_qr_code(prototype) for _ in range(repeats)]

    return content, prototype, dirty_qr_imgs


def normalize_image(image, target_size=(512, 512)):
    image = image.resize(target_size)

    return np.array(image.convert('L')).reshape((target_size[0], target_size[1], 1)) / 255.0


def load_qr_code_data(target_size, batch_size=32, batches_per_epoch=512, encoder=None, paired=False,
                      clean_image_every_n=8,
                      initial_noise=(1, 1, 1),
                      noisiest_epoch=60):
    """Infinite generator to create QR codes with progressively increasing noise, maxing out by the end of the epochs."""
    image_count = 0
    images_per_epoch = batches_per_epoch * batch_size  # Images per epoch

    while True:
        # Determine the current epoch and batch within the epoch
        current_epoch = image_count // images_per_epoch
        batch_in_epoch = (image_count % images_per_epoch) // 32

        if noisiest_epoch == 0:
            epoch_multiplier = 1
        else:
            # Calculate maximum noise level for the current epoch
            epoch_multiplier = min(current_epoch / (noisiest_epoch + 1e-8), 1)
        max_epoch_shift = epoch_multiplier * 20
        max_epoch_rotation = (epoch_multiplier * 180)
        max_epoch_noise_range = (epoch_multiplier * 100)

        # Calculate noise for the current batch within the epoch
        batch_multiplier = min(batch_in_epoch / (batches_per_epoch - 1), 1)  # Scales from 0 to 1 within each epoch
        max_shift = int(max_epoch_shift * batch_multiplier) + initial_noise[0]
        max_rotation = int(max_epoch_rotation * batch_multiplier) + initial_noise[1]
        noise_range = (
            int(0),
            int(max_epoch_noise_range * batch_multiplier) + initial_noise[2]
        )

        content, clean, dirty = generate_qr_code(repeats=1,
                                                 max_sequence_length=encoder.max_sequence_length)

        clean_img = normalize_image(clean, target_size)

        if image_count % clean_image_every_n == 0:
            dirty_img = clean_img
        else:
            # Apply noisier transformations based on current epoch and batch
            dirty_img = dirty_qr_code(clean, max_shift=max_shift, max_rotation=max_rotation, noise_range=noise_range)
            dirty_img = normalize_image(dirty_img, target_size)

        if paired:
            yield dirty_img, clean_img
        else:
            encoded_content = encoder.encode(content)
            yield dirty_img, encoded_content

        # Increment image count
        image_count += 1

        # Keep noise steady after the final epoch
        if image_count >= images_per_epoch * noisiest_epoch:
            image_count = images_per_epoch * (noisiest_epoch - 1)


def create_dataset(target_size=(512, 512), batch_size=32, shuffle=True, max_seq_len=512, num_chars=128, paired=False,
                   noisiest_epoch=0, batches_per_epoch=512):
    encoder = CharLevelEncoder(max_sequence_length=max_seq_len, num_chars=num_chars)  # Initialize encoder

    dataset = tf.data.Dataset.from_generator(
        lambda: load_qr_code_data(target_size, batch_size=batch_size, batches_per_epoch=batches_per_epoch,
                                  encoder=encoder, paired=paired,
                                  clean_image_every_n=8,
                                  initial_noise=(1, 1, 1), noisiest_epoch=noisiest_epoch
                                  ),
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32),  # X (dirty QR code)
            tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32) if paired else
            tf.TensorSpec(shape=(max_seq_len, num_chars), dtype=tf.float32)  # Y (clean QR code or encoded content)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=250)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    #    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    target_size = (256, 256)  # Example target size for images
    dataset = create_dataset(target_size=target_size, batch_size=32)

    for images, contents in dataset.take(1):  # Take one batch for example
        print("Images shape:", images.shape)
        print("Contents shape:", contents.shape)
        break
