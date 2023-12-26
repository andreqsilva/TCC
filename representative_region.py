import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def shannon_entropy(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    occurrence_porcent = histogram / np.sum(histogram) # número de pixels
    occurrence_porcent = occurrence_porcent[occurrence_porcent != 0] # evitar log(0)
    SE = -np.sum(occurrence_porcent * np.log2(occurrence_porcent))
    return SE

def get_best_region(image, rows, columns, grid_size):
    highest_entropy = 0
    for r in range(0, rows, grid_size[0]):
        r_end = r + grid_size[0]
        if r_end > rows:
            continue # ignora as regiões menores
        for c in range(0,  columns, grid_size[1]):
            c_end = c + grid_size[1]
            if c_end > columns:
                continue
            grid = image[r:r_end, c:c_end]
            SE = shannon_entropy(grid)
            if SE > highest_entropy:
                highest_entropy = SE
                selectd_region = grid

    return selectd_region, highest_entropy

def grid_parameters(rows, columns, magnification):
    lowest_dimension = min(rows, columns)
    Q = [(magnification / lowest_dimension * 100) * (rows / 100),
         (magnification / lowest_dimension * 100) * (columns / 100)]
    return math.floor(Q[0]), math.floor(Q[1])

def select_representative_region(image, rows, columns, magnification):
    grid_row, grid_column = grid_parameters(rows, columns, magnification)
    quadrant_size = (grid_row, grid_column)
    selected_region, highest_entropy = get_best_region(image, rows, columns, quadrant_size)

    '''# plotar região seleciona com seu histograma
    grayscale_image = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)    
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.5})

    # regigão selecionada
    axs[0].imshow(cv2.cvtColor(selected_region, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Região Selecionada')
    axs[0].axis('on')

    # histograma
    axs[1].plot(histogram, color='blue', alpha=0.7)
    axs[1].fill_between(range(256), histogram.flatten(), color='blue', alpha=0.3)
    axs[1].set_title('Histograma da Região Selecionada')
    axs[1].set_xlabel('Intensidade de Pixel')
    axs[1].set_ylabel('Número de Pixels')

    #plt.tight_layout()
    plt.show()'''

    return selected_region
