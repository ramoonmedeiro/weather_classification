import os
import cv2
import pandas as pd
import numpy as np
import csv

def join():
	os.system('ls *.jpg *jpeg >> datasets/imagens.txt')
	return

def resize_fig(shape):
	with open('../datasets/imagens.txt', 'r') as arquivo:
		conteudo = arquivo.readlines()
		for linha in conteudo:
			linha = linha.replace('\n', '')
			imagem = cv2.imread(linha)
			imagem = cv2.resize(imagem, shape)
			cv2.imwrite(linha, imagem)
	return

def create_labels():
	cloudy = {i: 'cloudy' for i in range(1, 301, 1)}
	rain = {i: 'rain' for i in range(301, 515, 1)}
	shine = {i: 'shine' for i in range(515, 767, 1)}
	sunrise = {i: 'sunrise' for i in range(767, 1124, 1)}

	labels = {**cloudy,**rain, **shine, **sunrise}

	with open('../datasets/labels.txt', 'w') as arquivo:
		arquivo.write('labels\n')
		for v in labels.values():
			arquivo.write(f'{v}\n')
		
   		
	return

def save_array():
	imagens = []

	with open('../datasets/imagens.txt', 'r') as arquivo:
		conteudo = arquivo.readlines()
		for linha in conteudo:
			linha = linha.strip()
			imagem = cv2.imread(f'../{linha}')
			# transformando BGR para RGB
			im_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
			imagens.append(im_rgb)

	np.save('../datasets/pixels.npy', np.array(imagens))

	return