import cv2

i = 1
with open('imagens-jpg.txt', 'r') as arquivo:
	conteudo = arquivo.readlines()
	for linha in conteudo:
		linha = linha.replace('\n', '')
		imagem = cv2.imread(f'/home/ramon/dataset2/{linha}')
		print(f'{i}: {imagem.shape}')
		i += 1