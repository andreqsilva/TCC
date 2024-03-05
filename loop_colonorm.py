import numpy as np
import pandas as pd
import cv2
import sys
import os
import shutil
import time

from image_similarity_measures.quality_metrics import ssim, psnr, fsim, uiq
from metrics import qssim
from stainsep import stainsep
from databases import MITOS, DISPLASIA, BUG

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
    database = "MITOS"

    for scheme in ['KL', 'Renyi', 'ED']:

        #scheme = "KL"
        magnification = 40

        make_dirs(database, scheme)
        targets, sources = MITOS("../Bases/MITOS")

        if len(targets) != len(sources):
            print("Número de imagens é imcompatível")
            sys.exit()

        nImages = len(targets)

        out_uiq = []
        out_fsim = []
        out_psnr = []
        out_qssim = []
        out_ssim = []

        print(f"\nDatabase: {database}\t Magnification: {magnification}x\t Scheme: {scheme}\t Total images: {nImages}\n")
        print(f"Image\t Filename\t\t Completed\t Estimated (h:m)")

        start = time.time()
        for index, (target_path, source_path) in enumerate(zip(targets, sources), start=1):
            # imagem alvo
            target_filename = target_path[(target_path.rfind("\\") + 1):]
            target = cv2.cvtColor(load_image(target_path), cv2.COLOR_BGR2RGB)
            [Wi, Hi] = stainsep(target, target_filename, database, magnification, nstains, scheme)

            # imagem original
            source_filename = source_path[(source_path.rfind("\\") + 1):]
            source = cv2.cvtColor(load_image(source_path), cv2.COLOR_BGR2RGB)
            [Wis, His] = stainsep(source, source_filename, database, magnification, nstains, scheme)

            # color nomalization
            our = SCN(source, Hi.T, Wi, His.T)

            #out_uiq.append(uiq(source, our))
            out_fsim.append(fsim(source, our))
            out_psnr.append(psnr(source, our))
            out_qssim.append(qssim(source, our))
            out_ssim.append(ssim(source, our))

            completed = round((index/nImages) * 100, 2)

            current_time = time.time() - start
            total_estimated_time = (current_time / index) * nImages - current_time
            hours_estimated_time = total_estimated_time / 3600 

            str_hours, str_minutes = str(hours_estimated_time).split('.')
            hours = int(str_hours)
            minutes = int(float("0." + str_minutes) * 60)

            print(f"{index}\t {source_filename}\t\t {completed}%\t\t {hours}:{minutes}")

        #save_metric(out_uiq, database, scheme, 'uiq')
        save_metric(out_fsim, database, scheme, 'fsim')
        save_metric(out_psnr, database, scheme, 'psnr')
        save_metric(out_qssim, database, scheme, 'qssim')
        save_metric(out_ssim, database, scheme, 'ssim')
    
if __name__ == "__main__":
    main()
