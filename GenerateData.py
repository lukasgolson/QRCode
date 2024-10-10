import numpy as np
import qrcode
import string
import os
import random
import tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageOps



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

    # make directory if not exist
    if not os.path.exists('data/images'):
        os.makedirs('data/images', exist_ok=True)

    if not os.path.exists('data/contents'):
        os.makedirs('data/contents', exist_ok=True)



    img = apply_random_shift(img, 25)

    img = apply_random_rotation(img, 45)

    img = add_random_noise(img)


#resize image to 512x512
    img = img.resize((resolution, resolution))

    img.save("data/images/" + name + '.png')

    with open("data/contents/" + name + '.txt', 'w') as f:
        f.write(content)


def add_random_noise(image, noise_factor_range=(1, 50)):
    """Add random pixel noise to the image with a randomized noise factor."""

    # Randomly select a noise factor within the provided range
    noise_factor = random.randint(noise_factor_range[0], noise_factor_range[1])

    # Convert image to a numpy array
    np_image = np.array(image)

    # Generate random noise
    noise = np.random.randint(-noise_factor, noise_factor, np_image.shape)

    # Add noise to the image and clip the values to be between 0 and 255
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values remain valid

    # Convert back to an image and return
    return Image.fromarray(noisy_image.astype('uint8'))


def apply_random_shift(image, max_shift=10):
    """Apply random shifts to the image using an affine transformation with a white background."""
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Create an affine transformation matrix to shift the image
    translation_matrix = (1, 0, shift_x, 0, 1, shift_y)

    # Create a new image with a white background (same size as the original)
    background = Image.new('L', image.size, color=255)  # 'L' for grayscale, 255 for white

    # Apply the transformation with an affine matrix
    shifted_image = image.transform(image.size, Image.AFFINE, translation_matrix, fillcolor=255)

    # Composite the shifted image over the white background
    background.paste(shifted_image)

    return background


def apply_random_rotation(image, max_angle=180):
    """Apply random rotation to the image."""
    angle = random.uniform(-max_angle, max_angle)  # Random angle between -30 and 30 degrees
    image = image.rotate(angle, expand=True, fillcolor=255)  # Rotate and expand to prevent cropping

    # Create a new image with a white background (same size as the original)
    background = Image.new('L', image.size, color=255)  # 'L' for grayscale, 255 for white

    background.paste(image)

    return background

def generate_data(population=None, max_length=300):
    # generate random data
    import random
    import string


    if population is None:
        population = string.ascii_uppercase + string.digits

    length = random.randint(1, max_length)

    content = ''.join(random.choices(population, k=length))

    return content


if __name__ == '__main__':
    # QR code count
    count = 1_500_000

    random.seed("Lukas G. Olson")


# a third should be letters only
    # a third should be numbers only
    # a third should be mixed
    for i in tqdm.tqdm(range(count), desc="Generating QR codes"):

        content = None


        if i % 4 == 0:
            # First quarter: less than 25 characters
            max_length = random.randint(1, 25)
        elif i % 4 == 1:
        # Second quarter: less than 100 characters
            max_length = random.randint(26, 100)
        elif i % 4 == 2:
        # Third quarter: less than 250 characters
            max_length = random.randint(101, 250)
        else:
            # Fourth quarter: less than 500 characters
            max_length = random.randint(251, 500)

        if i % 3 == 0:
            content = generate_data(string.ascii_uppercase, max_length)  # 2000)
        elif i % 3 == 1:
            content = generate_data(string.digits, max_length)  # 7089)
        else:
            content = generate_data(string.ascii_uppercase + string.digits, max_length)  # 4296)

        # generate QR code
        generate_qr(content, 'QR' + str(i))

# %%
