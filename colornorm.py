import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import os
import shutil

from image_similarity_measures.quality_metrics import ssim, psnr, fsim, uiq
from metrics import qssim

from stainsep import stainsep

def make_dirs(database, scheme):
    out_path = f"./out/{database}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    scheme_path = os.path.join(out_path, scheme)
    if not os.path.exists(scheme_path):
        os.makedirs(scheme_path)
        os.makedirs(os.path.join(scheme_path, "V"))
        shutil.copy("../hpcNMF/hpcNMF.win", os.path.join(scheme_path, "V"))
        os.makedirs(os.path.join(scheme_path, "W"))
        os.makedirs(os.path.join(scheme_path, "metrics"))

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

def save_metric(out_metric, database, scheme, metric_name):
    np.savetxt(f'./out/{database}/{scheme}/metrics/{metric_name}.txt', out_metric, fmt='%.4f')

def main():
    nstains = 2

    database = "Displasia"
    scheme = "KL"
    magnification = 40

    make_dirs(database, scheme)

    # imagem alvo
    target_filename = "DCIS (139)"   
    target_path = "../Bases/Reference images/DCIS (139).tif"
    target = cv2.cvtColor(load_image(target_path), cv2.COLOR_BGR2RGB)
    [Wi, Hi] = stainsep(target, target_filename, database, magnification, nstains, scheme)

    # imagem original
    source_filename = "image003-2-roi1"
    source_path = "../Bases/Displasia/ROIs_no_pre_processing/healthy/image003-2-roi1.tif"
    source = cv2.cvtColor(load_image(source_path), cv2.COLOR_BGR2RGB)
    [Wis, His] = stainsep(source, source_filename, database, magnification, nstains, scheme)

    # color nomalization
    our = SCN(source, Hi.T, Wi, His.T)

    #show_image(cv2.cvtColor(our, cv2.COLOR_RGB2BGR), 'Imagem normalizada')
    cv2.imwrite(f"./out/{database}/{scheme}/{source_filename}.png", cv2.cvtColor(our, cv2.COLOR_RGB2BGR))

    out_uiq = []
    out_fsim = []
    out_psnr = []
    out_qssim = []
    out_ssim = []

    out_uiq.append(uiq(source, our))
    out_fsim.append(fsim(source, our))
    out_psnr.append(psnr(source, our))
    out_qssim.append(qssim(source, our))
    out_ssim.append(ssim(source, our))

    save_metric(out_uiq, database, scheme, 'uiq')
    save_metric(out_fsim, database, scheme, 'fsim')
    save_metric(out_psnr, database, scheme, 'psnr')
    save_metric(out_qssim, database, scheme, 'qssim')
    save_metric(out_ssim, database, scheme, 'ssim')
    
    #cv2.imwrite(f"./results/Variacao de concentracao de corantes/{scheme}/{source_filename}.png", our)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), )
    images = [target, source, our]
    titles = ['Imagem de referência', 'Imagem original', 'Imagem normalizada']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    #show_image(our, f"{source_filename}")

    print(f"Imagem {source_filename} normalizada")

if __name__ == "__main__":
    main()
