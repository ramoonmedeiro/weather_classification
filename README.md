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
A rede neural é composta por uma camada de convolução, onde 32 kernels (matrizes 3x3 para esse caso) serão aplicados para gerar 32 mapas de características e posterior aplicação da função de ativação reLU para retirar números negativos na matriz resultante. Aplicou-se o Max Pooling nos mapas de características, tal aplicação é importante já que nesta etapa mais características relevantes podem ser extraídas e também pode-se evitar o overfitting ou ruídos. Posteriormente aplica-se o Flatten, para reordenar os dados para a camada de entrada da rede neur



