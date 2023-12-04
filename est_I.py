import numpy as np
import cv2
from BLTrans import beer_lambert
from representative_region import select_representative_region

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem em {path}.")
    return np.array(image)

def main():
    images = ['S13-92 A1-7 B', 'S13-93 A1-14 B', 'S13-95 A2-1 B']
    magnification = 40
    
    for image_name in images:
        path = f"./dataset/{image_name}.tif"
        #image_name = "S13-92 A1-7 B"
        image = load_image(path)
        
        I = select_representative_region(image, magnification)
        beer_lambert(I, image_name)

if __name__ == "__main__":
    main()