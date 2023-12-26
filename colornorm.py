import numpy as np
import cv2

from stainsep import stainsep

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem em {path}.")
    return np.array(image)

def main():
    magnification = 40
    
    source_path = "S13-92 A1-7 B.tif"
    target_path = "S13-93 A1-14 B.tif"
    
    source = load_image(source_path)
    target = load_image(target_path)

    Wis, His = stainsep(source, "S13-92 A1-7 B", magnification)
    Wi, Hi = stainsep(target, "S13-93 A1-14", magnification)

if __name__ == "__main__":
    main()