import numpy as np
import qrcode
import string
import os
import random
import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_qr(content, name, error_correction=qrcode.constants.ERROR_CORRECT_L, resolution=512):
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=10,
        border=4,
    )

    qr.add_data(content)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Make directory if it doesn't exist
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/contents', exist_ok=True)

    img = apply_random_shift(img, 30)
    img = apply_random_rotation(img, 60)
    img = add_random_noise(img)

    # Resize image to 512x512
    img = img.resize((resolution, resolution))

    img.save(f"data/images/{name}.png")

    with open(f"data/contents/{name}.txt", 'w') as f:
        f.write(content)


def add_random_noise(image, noise_factor_range=(1, 75)):
    """Add random pixel noise to the image with a randomized noise factor."""
    noise_factor = random.randint(noise_factor_range[0], noise_factor_range[1])
    np_image = np.array(image)
    noise = np.random.randint(-noise_factor, noise_factor, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values remain valid
    return Image.fromarray(noisy_image.astype('uint8'))


def apply_random_shift(image, max_shift=10):
    """Apply random shifts to the image using an affine transformation with a white background."""
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    translation_matrix = (1, 0, shift_x, 0, 1, shift_y)
    background = Image.new('L', image.size, color=255)
    shifted_image = image.transform(image.size, Image.AFFINE, translation_matrix, fillcolor=255)
    background.paste(shifted_image)
    return background


def apply_random_rotation(image, max_angle=180):
    """Apply random rotation to the image."""
    angle = random.uniform(-max_angle, max_angle)
    image = image.rotate(angle, expand=True, fillcolor=255)
    background = Image.new('L', image.size, color=255)
    background.paste(image)
    return background


def generate_data(population=None, max_length=300):
    if population is None:
        population = string.ascii_uppercase + string.digits
    length = random.randint(1, max_length)
    content = ''.join(random.choices(population, k=length))
    return content


def generate_qr_code(i):
    """Generate a QR code and return its index for reference."""
    if i % 4 == 0:
        max_length = random.randint(1, 25)
    elif i % 4 == 1:
        max_length = random.randint(26, 100)
    elif i % 4 == 2:
        max_length = random.randint(101, 250)
    else:
        max_length = random.randint(251, 500)

    if i % 3 == 0:
        content = generate_data(string.ascii_uppercase, max_length)
    elif i % 3 == 1:
        content = generate_data(string.digits, max_length)
    else:
        content = generate_data(string.ascii_uppercase + string.digits, max_length)

    generate_qr(content, f'QR{i}')
    return i


if __name__ == '__main__':
    count = 100_000
    random.seed("Lukas G. Olson")

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(generate_qr_code, i) for i in range(count)]
        for future in tqdm.tqdm(as_completed(futures), total=count, desc="Generating QR codes"):
            future.result()  # This will raise any exceptions that occurred during execution
