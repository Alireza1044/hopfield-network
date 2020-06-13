import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import re

errors_10_16 = []
errors_30_16 = []
errors_60_16 = []

errors_10_32 = []
errors_30_32 = []
errors_60_32 = []

errors_10_64 = []
errors_30_64 = []
errors_60_64 = []


def draw_table():
    table_data = []

    table_data.append(['16', str(round(np.average(errors_10_16), 2)) + ' %',
                       str(round(np.average(errors_30_16), 2)) + ' %',
                       str(round(np.average(errors_60_16), 2)) + ' %'])
    table_data.append(['32', str(round(np.average(errors_10_32), 2)) + ' %',
                       str(round(np.average(errors_30_32), 2)) + ' %',
                       str(round(np.average(errors_60_32), 2)) + ' %'])
    table_data.append(['64', str(round(np.average(errors_10_64), 2)) + ' %',
                       str(round(np.average(errors_30_64), 2)) + ' %',
                       str(round(np.average(errors_60_64), 2)) + ' %'])

    fig, ax = plt.subplots()
    table = ax.table(cellText=table_data, cellLoc='center'
                     , colLabels=['Font Size', '10% Noise', '30% Noise', '60% Noise'])
    table.set_fontsize(15)
    table.scale(3, 3)
    fig.canvas.draw()
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.from_extents(bbox.xmin - 5, bbox.ymin - 5, bbox.xmax + 5, bbox.ymax + 5)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    ax.axis('off')
    fig.savefig('error_table.png', bbox_inches=bbox_inches)
    plt.show()


def read_image_dir(folder, name, size):
    path = name + size + '.bmp'
    img = Image.open(folder + path).convert(mode="L")
    imgArray = np.asarray(img, dtype=np.uint8).flatten()
    x = np.zeros(imgArray.shape, dtype=np.float)
    x[imgArray > 60] = 1
    x[x == 0] = 0
    return x


def read_image_path(path, folder):
    img = Image.open(folder + path).convert(mode="L")
    imgArray = np.asarray(img, dtype=np.uint8).flatten()
    x = np.zeros(imgArray.shape, dtype=np.float)
    x[imgArray > 60] = 1
    x[x == 0] = 0
    return x


def add_to_array(size, error, data):
    if error == '10':
        if size == '16':
            errors_10_16.append(data)
        elif size == '32':
            errors_10_32.append(data)
        elif size == '64':
            errors_10_64.append(data)
    elif error == '30':
        if size == '16':
            errors_30_16.append(data)
        elif size == '32':
            errors_30_32.append(data)
        elif size == '64':
            errors_30_64.append(data)
    elif error == '60':
        if size == '16':
            errors_60_16.append(data)
        elif size == '32':
            errors_60_32.append(data)
        elif size == '64':
            errors_60_64.append(data)


if __name__ == '__main__':
    dir = os.listdir('images')
    dir.remove('.DS_Store')

    dir_noise = os.listdir('results')
    dir_noise.remove('.DS_Store')

    for photo in dir_noise:
        error, name = re.split('_|16|32|64|.bmp', photo)[0:2]
        size = re.split('_|A|B|C|D|E|F|G|H|I|J|.bmp', photo)[2]

        img = read_image_dir('images/', name, size)
        img_noise = read_image_path(path=photo, folder='noisy_images/')
        add_to_array(size, error, 100 * (1 - np.average(np.abs(img - img_noise))))

    draw_table()
