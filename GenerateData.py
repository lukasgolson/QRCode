import qrcode
import string
import os
import random


def generate_qr(content, name, error_correction=qrcode.constants.ERROR_CORRECT_L):
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=10,
        border=2,
    )

    qr.add_data(content)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # make directory if not exist
    if not os.path.exists('data/images'):
        os.makedirs('data/images', exist_ok=True)

    if not os.path.exists('data/contents'):
        os.makedirs('data/contents', exist_ok=True)

    #resize image to 512x512
    img = img.resize((512, 512))

    img.save("data/images/" + name + '.png')

    with open("data/contents/" + name + '.txt', 'w') as f:
        f.write(content)


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
    count = 2000

    random.seed("Lukas G. Olson")


# a third should be letters only
    # a third should be numbers only
    # a third should be mixed
    for i in range(count):

        content = None

        if i % 3 == 0:
            content = generate_data(string.ascii_uppercase, 500)  # 2000)
        elif i % 3 == 1:
            content = generate_data(string.digits, 500)  # 7089)
        else:
            content = generate_data(string.ascii_uppercase + string.digits, 500)  # 4296)

        # generate QR code
        generate_qr(content, 'QR' + str(i))

# %%
