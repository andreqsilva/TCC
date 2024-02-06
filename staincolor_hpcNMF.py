import subprocess
import numpy as np
import sys

def normalize(W):
    for i in range(W.shape[1]):
        # normaliza cada coluna de W pelo comprimento Euclidiano (norma L2)
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i], 2)

    #W[0], W[-1] = W[-1].copy(), W[0].copy()
    return np.abs(W)

def estW(path):
    W = []
    for i in range(2):
        w_path = path[:-1] + str(i)
        with open(w_path, "r") as file:
            rounds = []
            file.seek(0) # volta para o início do arquivo
            lines = file.readlines()
            for line in lines[1:]:
                new_line = line.strip().split()[1:]
                new_line = [float(value) for value in new_line]
                rounds.append(new_line)
            W.append(np.mean(rounds, axis=0))
    return normalize(np.array(np.transpose(W)))

def get_staincolor_hpcNMF(scheme, filename, dir, file):
    try:
        matrix = f"{filename}.txt"
        command = [dir + file] + ["-s", scheme, "-i", matrix]
        print(f"Gerando a matriz W de {filename}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=dir)
        if result.returncode == 0:
            Wi = estW(f"./tests/Variacao de concentracao de corantes/{scheme}/{filename}.{scheme}.k2.W0")
            print("Matriz W gerada com sucesso!")
            return Wi
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("stderr:", e.stderr)
        print("returncode:", e.returncode)
        raise Exception(f"Error: {e}")