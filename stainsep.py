import cv2
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import NMF

from representative_region import select_representative_region
from staincolor_hpcNMF import get_staincolor_hpcNMF, estW

def show_image(image, image_name):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

def BLTrans(I):
    Ivecd = np.double(I.reshape((I.shape[0] * I.shape[1], I.shape[2])))
    # matriz de densidade óptica
    V = np.log(255) - np.log(Ivecd + 1) # V=WH, +1 is to avoid divide by zero

    #V[V < 0] = 0 

    # V com exclusão de pixels brancos
    C = cv2.cvtColor(I, cv2.COLOR_BGR2Lab) # conversão do RGB para o LAB
    luminlayer = C[:, :, 0] # extrai o primeiro canal (Luminância)

    # threshold = 0.9
    validpoints = (luminlayer / 255) < 0.9
    Inew = Ivecd[validpoints.flatten(), :]

    VforW = np.log(255) - np.log(Inew + 1)

    return np.transpose(V), np.transpose(VforW)

def estH(V, Ws, nrows, ncolumns, nstains):
    # calcula a pseudo-inversa não-negativa
    Hs_vec = np.linalg.pinv(Ws) @ V
    Hs_vec[Hs_vec < 0] = 0  # transforma valores negativos em 0
    
    #Hs = np.reshape(Hs_vec, (nrows, ncolumns, nstains))

    Irecon = np.dot(Hs_vec.T, Ws.T)
    Irecon = (255 * np.exp(-Irecon)).reshape((nrows, ncolumns, 3)).astype(np.uint8)

    return Hs_vec, Irecon

def stainsep(I, filename, magnification, nstains, scheme):
    # verifica se a imagem é colorida (possui 3 dimensões)
    ndimI = len(I.shape)
    if ndimI != 3:
        print("A imagem não possui 3 dimensões")
        sys.exit(1)

    [rows, columns] = I.shape[:2]
    
    # selecionar região representativa
    region = select_representative_region(I, rows, columns, magnification)

    # transformada de Beer-Lamber
    [V, V1] = BLTrans(region)
    
    # salva a matriz V no formato txt
    df = pd.DataFrame(V, index=['R', 'G', 'B'], columns=list(range(0, V.shape[1])))
    df.to_csv(f'./tests/Variacao de concentracao de corantes/{scheme}/{filename}.txt', sep='\t', header='\t')
    
    Wi = get_staincolor_hpcNMF(scheme, filename, f"./tests/Variacao de concentracao de corantes/{scheme}/", "hpcNMF.win")
    #Wi = estW(f"./tests/Variacao de concentracao de corantes/{scheme}/{filename}.{scheme}.k2.W0")
    
    # np.savetxt(f'./tests/Variacao de concentracao de corantes/{scheme}/V_{filename}.txt', W)

    #model = NMF(n_components=nstains, init='random', random_state=42, max_iter=1200)
    #Wi = model.fit_transform(V)

    [BLTI, BLTI1] = BLTrans(I)
    [Hi, Irec] = estH(BLTI, Wi, rows, columns, nstains)

    #Hi = np.linalg.lstsq(Wi, np.reshape(BLTI, (3, rows * columns)), rcond=None)[0]
    #Hi = np.reshape(Hi, (rows, columns, nstains))

    return Wi, Hi
