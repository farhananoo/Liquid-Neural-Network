import numpy as np
from skimage import color


def rgb_to_lab(tile):
    lab = color.rgb2lab(tile)
    return lab


def lab_to_rgb(lab):
    newtile = (color.lab2rgb(lab) * 255).astype(np.uint8)
    return newtile


def normalize_tile(tile, norm_vec):
    lab = rgb_to_lab(tile)
    tile_mean = [0, 0, 0]
    tile_std = [1, 1, 1]
    new_mean = norm_vec[0:3]
    new_std = norm_vec[3:6]
    for i in range(3):
        tile_mean[i] = np.mean(lab[:, :, i])
        tile_std[i] = np.std(lab[:, :, i])
        tmp = ((lab[:, :, i] - tile_mean[i]) * (new_std[i] / tile_std[i])) + new_mean[i]
        if i == 0:
            tmp[tmp < 0] = 0
            tmp[tmp > 100] = 100
            lab[:, :, i] = tmp
        else:
            tmp[tmp < -128] = 128
            tmp[tmp > 127] = 127
            lab[:, :, i] = tmp
    tile = lab_to_rgb(lab)
    return tile
