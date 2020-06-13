import numpy as np
from PIL import Image
import os

dim = 40
network_dim = dim ** 2


class Hopfield:
    def __init__(self):
        self.weight_matrix = np.zeros((network_dim, network_dim))

    def update_weights(self, input):
        for i in range(len(input)):
            for j in range(len(input)):
                if (i == j):
                    self.weight_matrix[i, j] = -50
                    break
                self.weight_matrix[i, j] += input[i] * input[j]
                self.weight_matrix[j, i] = self.weight_matrix[i, j]

    def train(self, dir):
        for photo_path in dir:
            image_matrix, image_size = read_image(photo_path, 'images/')
            image_matrix = image_matrix.flatten()
            self.update_weights(image_matrix)

    def restore(self, input):
        output = []
        for row in range(len(self.weight_matrix)):
            temp = 0
            for column in range(len(self.weight_matrix)):
                if row == column:
                    continue
                temp += self.weight_matrix[row, column] * input[column]
            output.append(sign(temp))
        return np.array(output, dtype='uint8')

    def feed(self, dir):
        for photo_path in dir:
            image_matrix, image_size = read_image(photo_path, 'noisy_images/')
            image_matrix = image_matrix.flatten()
            output = self.restore(image_matrix)
            output = np.reshape(output, (dim, dim))
            output = Image.fromarray(output, mode='L').resize(image_size)
            output.save('results/' + photo_path)


def sign(a):
    if a >= 0:
        return 255
    else:
        return 0


def read_image(path, folder):
    img = Image.open(folder + path).convert(mode="L")
    size = img.size
    img = img.resize((dim, dim))
    imgArray = np.asarray(img, dtype=np.uint8)
    x = np.zeros(imgArray.shape, dtype=np.float)
    x[imgArray > 60] = 1
    x[x == 0] = -1
    return x, size


def evaluate_outputs():
    return 1


if __name__ == '__main__':
    hopfield = Hopfield()
    dir = os.listdir('images')
    dir.remove('.DS_Store')
    hopfield.train(dir)

    dir = os.listdir('noisy_images')
    dir.remove('.DS_Store')
    hopfield.feed(dir)
