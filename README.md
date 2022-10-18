# Objetivos

Utilizar as redes neurais convolucionais (CNNs) para a classificação de imagens meteorológicas. Além disso, testarei as CNNs com e sem o data augmentation
a fim de estudo e comparações entre os modelos.

# Dataset

O dataset utilizado foi retirado do site <a href="https://data.mendeley.com/datasets/4drtyfjtfy/1">Weather dataset</a>, onde originalmente existem 1125 imagens de 4 classes, sendo elas: cloudy (nublado), rain (chuvoso), shine (ensolarado) e sunrise (Nascer do sol). Duas das imagens deste dataset eram gif's e foram excluídas, sendo elas da classe shine e rain. Então ao total são 1123 imagens separadas em: 300 da classe cloudy, 214 da classe rain, 
252 da classe shine e 357 da classe sunrise.

Os scripts <i>main.py</i> e <i>images.py</i> possuem funções que realizaram a extração dos pixels de cada imagem. Cada imagem possui dimensões diferentes, o que poderia ser um incômodo, então todas as imagens foram transformadas para terem 150x150 pixels e 3 canais de cores RGB. Além disso, houve também o tratamento e salvamento dos labels.

Com isso, segue as etapas de contrução da rede neural.

# Etapa de treinamento

Bibliotecas necessárias:

```
# Libs para manipulação e plotagem de dados
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Libs do Keras
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Libs do sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# Lib do scikeras
from scikeras.wrappers import KerasClassifier
```

Carregando o dataset:

```
data = np.load('/content/drive/MyDrive/Colab Notebooks/Datasets/pixels.npy')
labels = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/labels.txt', sep=',')
```
O conjunto de labels precisa ser transformado com a transformação do one hot encoder:

```
labels = labels['labels']
# 4 classes na camada de saída
labels = to_categorical(labels, 4)
```

Separação entre treino e teste. Além da normalização dos pixels:

```
X_treino, X_teste, y_treino, y_teste = train_test_split(data, labels, test_size=0.25, random_state=99)

# Tranformando valores para float32
X_treino = X_treino.astype('float32')
X_teste = X_teste.astype('float32')

# Normalizando os valores dos pixels
X_treino /= 255.0
X_teste /= 255.0
```
** Método sem augmentation **

Abaixo é apresentado os resultados das acurácias que as CNNs obtiveram sem o método de data augmentation. Inicia-se a criação da CNN para a validação cruzada:

```
def cnn_create():
  model = Sequential()
  model.add(Conv2D(32, (3,3), input_shape = (150, 150, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(units = 128, activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units = 4, activation = 'softmax'))
  model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model
```
A rede neural é composta por uma camada de convolução, onde 32 kernels (matrizes 3x3 para esse caso) serão aplicados para gerar 32 mapas de características e posterior aplicação da função de ativação reLU para retirar números negativos na matriz resultante. Aplicou-se o Max Pooling nos mapas de características, tal aplicação é importante já que nesta etapa mais características relevantes podem ser extraídas e também pode-se evitar o overfitting ou ruídos. Posteriormente aplica-se o Flatten, para reordenar os dados para a camada de entrada para a camada densa. Apenas uma camda oculta foi adicionada à rede neural, com 128 neurônios e função de ativação reLU. Utilizou uma camda de Dropout para reiniciar os pesos de 20% dos neurônios. A camada de saída possui 4 neurônios, já que o problema apresenta 4 classes. Abaixo cria-se um classificador com o scikeras:

```
# Classificador
clf = KerasClassifier(build_fn = cnn_create, epochs = 5, batch_size = 100)
# Validação Cruzada
results = cross_val_score(clf, X_treino, y_treino, cv = 10)

# Acurácia e desvio padrão
acc = results.mean()
std = results.std()

print(f'acc: {acc:.2f} +- {std:.2f}')

acc: 0.86 +- 0.04
```

A validação cruzada forneceu 86 % de acurácia para o conjunto de treinamento sem data augmentation.

** Com data augmentation **

Novamente cria-se a rede neural, da mesma forma que a anterior, porém, agora iremos utilizar o data augmentation:

```
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (150, 150, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 4, activation = 'softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Criando mais imagens no conjunto de treino
gtreino = ImageDataGenerator(rotation_range=7, horizontal_flip = True, shear_range = 0.3, height_shift_range=0.07, zoom_range=0.2)
gteste = ImageDataGenerator()

btreino = gtreino.flow(X_treino, y_treino, batch_size=128)
bteste = gteste.flow(X_teste, y_teste, batch_size=128)
```

Treinando o modelo com mais imagens e e resultado da acurácia:

```
model.fit_generator(btreino, steps_per_epoch = 842/128, epochs = 10, validation_data=bteste)
loss, acc = model.evaluate(X_treino, y_treino)
round(acc, 2)

0.95
```

Obteve-se uma acurácia de 95 %, que é maior do que o modelo sem data augmentation (~86 %). Agora iremos utilizar este modelo para aferir a acurácia para o modelo de testes:

```
resultados = model.predict(X_teste)
resultados = resultados > 0.5
```
