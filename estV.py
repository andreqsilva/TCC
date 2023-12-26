import numpy as np
import cv2
from hpcNMF.estW import calculate_W

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem em {path}.")
    return np.array(image)

def calculate_H(Ws, Ivecd):
    # calcula a pseudo-inversa
    Hs_vec = np.linalg.inv(Ws.T @ Ws) @ Ws.T @ Ivecd
    Hs_vec[Hs_vec < 0] = 0  # transforma valores negativos em 0
    return Hs_vec

def reconstruct_image(W, H, rows, columns, channels):
    V = np.dot(W, H).reshape((rows, columns, channels))
    V = np.clip(V, 0, 255).astype(np.uint8)
    return V

def show_image(image, image_name):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

def main():
    images = ['S13-92 A1-7 B', 'S13-93 A1-14 B', 'S13-95 A2-1 B']

    target_image_name = images[0]
    target_image_path = f"./dataset/{target_image_name}.tif"
    target_image = load_image(target_image_path)

    target_rows, target_columns, channels = target_image.shape
    Ivecd = np.double(target_image.reshape((channels, target_rows * target_columns)))
    
    path_w = f"./tests/ED/V-{images[0]}.ED.k2.W0"
    W_target = calculate_W(path_w)
    
    H_target = calculate_H(W_target, Ivecd)
    
    for source_image_name in images[1:]:
        source_image_path = f"./dataset/{source_image_name}.tif"
        source_image = load_image(source_image_path)
        
        source_rows, source_columns = source_image.shape[:2]
        source_Ivecd = np.double(source_image.reshape((channels, source_rows * source_columns)))
        
        path_W = f"./tests/ED/V-{source_image_name}.ED.k2.W0"
        W_source = calculate_W(path_W)
        H_source = calculate_H(W_source, source_Ivecd)
        
        V = reconstruct_image(W_target, H_source, source_rows, source_columns, channels)
        
        show_image(V, f'{source_image_name} reconstruida')
        cv2.imwrite(f"{source_image_name}_ED_reconstructed.png", V)

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()