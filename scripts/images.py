import os
import cv2
import pandas as pd
import numpy as np
import csv

def join():
	os.system('ls *.jpg *jpeg >> imagens.txt')
	return

def resize_fig(shape):
	with open('imagens.txt', 'r') as arquivo:
		conteudo = arquivo.readlines()
		for linha in conteudo:
			linha = linha.replace('\n', '')
			imagem = cv2.imread(linha)
			imagem = cv2.resize(imagem, shape)
			cv2.imwrite(linha, imagem)
	return

def create_labels():
	cloudy = ['cloudy' for i in range(300)]
	rain = ['rain' for i in range(214)]
	shine = ['shine' for i in range(252)]
	sunrise = ['sunrise' for i in range(357)]

	labels = cloudy+rain+shine+sunrise

	with open('labels.csv', 'w', newline='') as arquivo:
    	writer = csv.writer(arquivo)
    	writer.writerow(labels)

    return