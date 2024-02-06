import numpy as np
import os
import cv2
import math

def set_intersection_makers(image, image_size, grid_size):
    for r in range(0, image_size[0], grid_size[0] + 1):
        r_end = r + grid_size[0]
        if r_end > image_size[0]:
            continue
        for c in range(0, image_size[1], grid_size[1] + 1):
            c_end = c + grid_size[1]
            if c_end > image_size[1]:
                continue
            center_coordinates = (c_end, r_end)
            radius = 3
            color = (0, 255, 0)  # BGR (verde)
            # desenha o círculo na imagem
            cv2.circle(image, center_coordinates, radius, color, thickness=-1)
    return image

def image_grids(image, rows, columns, grid_size):
    vertical_images_vector = []
    for r in range(0, rows, grid_size[0]):
        r_end = r + grid_size[0]
        if r_end > rows:
            r_end = rows
        horizontal_images_vector = []
        for c in range(0, columns, grid_size[1]):
            c_end = c + grid_size[1]
            if c_end > columns:
                c_end = columns
            grid = image[r:r_end, c:c_end]
            if c_end != columns:
                horizontal_grid_line = np.zeros((grid.shape[0], 1, 3), dtype=np.uint8)
                grid = np.concatenate((grid, horizontal_grid_line), axis=1)
            horizontal_images_vector.append(grid)

        # concatenação horizontal para gerar um dos trechos da imagem
        horizontal_image = cv2.hconcat(horizontal_images_vector)
        if r_end != rows:
            vertical_image_line = np.zeros((1, horizontal_image.shape[1], 3), dtype=np.uint8)
            horizontal_image = np.concatenate((horizontal_image, vertical_image_line), axis=0)
        vertical_images_vector.append(horizontal_image)

    # concatenação vertical para unir os trechos horizontais
    final_image = cv2.vconcat(vertical_images_vector)
    final_image = set_intersection_makers(final_image, final_image.shape[:2], grid_size)
    return final_image

def get_grid_parameters(rows, columns, magnification):
    lowest_dimension = min(rows, columns)
    Q = [(magnification / lowest_dimension * 100) * (rows / 100),
         (magnification / lowest_dimension * 100) * (columns / 100)]
    return math.floor(Q[0]), math.floor(Q[1])

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem em {path}.")
    return np.array(image)    

def list_images(dir):
    try:
        files = os.listdir(dir)
        files = [file for file in files if os.path.isfile(os.path.join(dir, file))]   
        return files
    except Exception as e:
        print(f"Erro ao listar arquivos do diretório {dir}: {e}")

def grid_generation(images, dir, magnification):
    target_dir = "./grid-images/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in images:
        path = os.path.join(dir, file)
        I = load_image(path)
        [n_rows, n_columns] = I.shape[:2]
        [grid_row, grid_column] = get_grid_parameters(n_rows, n_columns, magnification)
        image_in_grids = image_grids(I, n_rows, n_columns, [grid_row, grid_column])
        new_file = "grid_" + os.path.splitext(file)[0] + ".png" 
        target_path = os.path.join(target_dir, new_file)
        cv2.imwrite(target_path, image_in_grids)

def main():
    dir = "./dataset/"
    #images = list_images(dir)
    images = ['S13-92 A1-7 B.tif']
    magnification = 40
    grid_generation(images, dir, magnification)

if __name__ == "__main__":
    main()