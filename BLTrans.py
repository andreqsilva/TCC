import numpy as np

def beer_lambert(image, image_name):
    n_rows, n_columns, n_channels = image.shape
    Ivecd = np.double(image.reshape((n_rows * n_columns, n_channels)))

    # V=WH, +1 is to avoid divide by zero
    # matriz de densidade óptica
    V = np.log(255) - np.log(Ivecd + 1)

    # transposta da matriz
    V = np.transpose(V)
    np.savetxt(f'./tests/ED/V-{image_name}.txt', V)