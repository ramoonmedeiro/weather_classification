import pandas as pd
import numpy as np
import csv


cloudy = ['cloudy' for i in range(300)]
rain = ['rain' for i in range(214)]
shine = ['shine' for i in range(252)]
sunrise = ['sunrise' for i in range(357)]

labels = cloudy+rain+shine+sunrise

with open('labels.csv', 'w', newline='') as arquivo:
    writer = csv.writer(arquivo)
    writer.writerow(labels)