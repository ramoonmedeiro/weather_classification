import os
import cv2

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