import cv2
import numpy as np

path = '/home/ramon/dataset2/datasets/imagens.txt'

imagens = []

with open(path, 'r') as arquivo:
    conteudo = arquivo.readlines()
    for linha in conteudo:
        linha = linha.strip()
        imagem = cv2.imread(f'../{linha}')
        imagens.append(imagem)

a = np.array(imagens)
print(a.shape)   