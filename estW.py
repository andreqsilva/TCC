import numpy as np

def calculate_W(path):
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
    return np.array(np.transpose(W))