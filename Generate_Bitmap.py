from PIL import Image, ImageFont
import numpy as np
import random
import cv2
import os

font_sizes = [16, 32, 64]
noises = [0.1, 0.3, 0.6]


def add_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def generate_images():
    for font_size in font_sizes:
        font = ImageFont.truetype('Tahoma.ttf', font_size)
        for char in 'ABCDEFGHIJ':
            im = Image.Image()._new(font.getmask(char))
            im.save('images/' + char + str(font_size) + '.bmp')


if __name__ == '__main__':
    generate_images()

    dir = os.listdir('images')
    dir.remove('.DS_Store')
    for noise in noises:
        for photo_path in dir:
            image = cv2.imread('images/' + photo_path, 0)  # Only for grayscale image
            noise_img = add_noise(image, noise)
            cv2.imwrite('noisy_images/' + str(int(noise*100)) + '_' + photo_path, noise_img)

