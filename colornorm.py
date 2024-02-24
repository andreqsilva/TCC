import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

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
    #Hso = np.reshape(Hso, (Hso.shape[0] * Hso.shape[1], Hso.shape[2]))
    Hso_Rmax = np.percentile(Hso.flatten(), 99) # percentil de 95
    #Hta = np.reshape(Hta, (Hta.shape[0] * Hta.shape[1], Hta.shape[2]))
    Hta_Rmax = np.percentile(Hta.flatten(), 99)
    normfac = Hta_Rmax / Hso_Rmax # fator de normalização
    Hsonorm = Hso * np.tile(normfac, (Hso.shape[0], 1))

    Ihat = np.dot(Wta, Hsonorm.T)

    # Back projection into spatial intensity space (Inverse Beer-Lambert space)
    sourcenorm = (255 * np.exp(-np.reshape(Ihat.T, source.shape))).astype(np.uint8)

    return sourcenorm

def main():
    nstains = 2
    magnification = 40
    scheme = "KL"

    # imagem alvo
    target_filename = "ilu_47453_01_03"
    target_path = f"./dataset/Variacao de concentracao de corantes/refs/{target_filename}.tif"
    target = cv2.cvtColor(load_image(target_path), cv2.COLOR_BGR2RGB)
    [Wi, Hi] = stainsep(target, target_filename, magnification, nstains, scheme)

    # imagem original
    source_filename = "ilu_47453_01_01"   
    source_path = f"./dataset/Variacao de concentracao de corantes/{source_filename}.tif"
    source = cv2.cvtColor(load_image(source_path), cv2.COLOR_BGR2RGB)
    [Wis, His] = stainsep(source, source_filename, magnification, nstains, scheme)

    # color nomalization
    our = SCN(source, Hi.T, Wi, His.T)

    #show_image(cv2.cvtColor(our, cv2.COLOR_RGB2BGR), 'Imagem normalizada')
    #cv2.imwrite(f"./results/Variacao de concentracao de corantes/{scheme}/{source_filename}.png", cv2.cvtColor(our, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"./results/Variacao de concentracao de corantes/{scheme}/{source_filename}.png", our)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), )
    images = [target, source, our]
    titles = ['Imagem de referência', 'Imagem original', 'Imagem normalizada']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    #show_image(our, f"{source_filename}")

if __name__ == "__main__":
    main()