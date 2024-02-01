import numpy as np
import cv2

from stainsep import stainsep

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem em {path}.")
    return np.array(image)

def show_image(image, image_name):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

def SCN(source, Hta, Wta, Hso):
    Hso = np.reshape(Hso, (Hso.shape[0] * Hso.shape[1], Hso.shape[2]))
    Hso_Rmax = np.percentile(Hso.flatten(), 99) # percentil de 95

    Hta = np.reshape(Hta, (Hta.shape[0] * Hta.shape[1], Hta.shape[2]))
    Hta_Rmax = np.percentile(Hta.flatten(), 99)

    # fator de normalização
    normfac = Hta_Rmax / Hso_Rmax

    # multiplica Hso pelo fator de normalização ao longo do eixo 0
    Hsonorm = Hso * np.tile(normfac, (Hso.shape[0], 1))
    #Hsonorm = Hso * normfac

    Ihat = np.dot(Wta, Hsonorm.T)

    # Back projection into spatial intensity space (Inverse Beer-Lambert space)
    sourcenorm = (255 * np.exp(-Ihat.T.reshape(source.shape))).astype(np.uint8)
    return sourcenorm

def reconstruct_image(W, H, rows, columns, channels):
    H = np.reshape(H, (H.shape[2], H.shape[0] * H.shape[1]))
    V = np.dot(W, H).reshape((rows, columns, channels))
    #return (255 * np.exp(-np.reshape(V, (rows, columns, channels))))
    V = np.clip(V, 0, 255).astype(np.uint8)
    return V

def main():
    nstains = 2
    magnification = 40
    scheme = "KL"

    # imagem alvo
    target_filename = "S13-93 A1-14 B"
    target_path = f"./dataset/{target_filename}.tif"
    target = load_image(target_path)
    [Wi, Hi] = stainsep(target, target_filename, magnification, nstains, scheme)

    source_filename = "S13-92 A1-7 B"   
    source_path = f"./dataset/{source_filename}.tif"
    source = load_image(source_path)
    [Wis, His] = stainsep(source, source_filename, magnification, nstains, scheme)

    # color nomalization
    our = SCN(source, Hi, Wi, His)

    #our = reconstruct_image(Wi, Hi, source.shape[0], source.shape[1], source.shape[2])

    show_image(our, "Imagem Reconstruida")

if __name__ == "__main__":
    main()