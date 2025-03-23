#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:37:43 2025

@author: vitoriamariana
"""

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
#%%
#/Users/vitoriamariana/Desktop/Faculdade/Polytech/S8-2025/Traitement-Statistiques-donnees/TP03/INFARCTU.xlsx
# On charge les données
infarto = pd.read_excel("/Users/vitoriamariana/Desktop/Faculdade/Polytech/S8-2025/Traitement-Statistiques-donnees/TP03/INFARCTU.xlsx",sheet_name=0,header=0,index_col=0)
infarto.info()
[lines,columns] = infarto.shape #n=notes.shape[0]
individus_inconnu=infarto.iloc[102:107]
individus_inconnu = individus_inconnu.drop('PRONO', axis=1)
infarto_connu=infarto[0:101]

#%%
infarto_connu_variable=infarto_connu.iloc[:,1:-1]
#%%
infarto_connu_variable.boxplot(by="PRONO")
infarto_connu_variable.groupby('PRONO').mean()
#%% Matrice en utilisant la heatmap de seaborn
pd.plotting.scatter_matrix(infarto_connu.iloc[:, 1:-1], c=infarto_connu['C'])
#%%
infarto_connu_variable = infarto_connu.drop('PRONO', axis=1)
lda = LinearDiscriminantAnalysis()
infarto_transformee = lda.fit_transform(infarto_connu_variable, infarto_connu_variable['C'])
#%%
# Visualizando os componentes discriminantes
plt.figure(figsize=(8, 6))
plt.scatter(infarto_transformee[:, 0], np.zeros_like(infarto_transformee[:, 0]), c=infarto_connu_variable['C'], cmap='viridis')
plt.title("Transformação LDA por grupo 'C'")
plt.xlabel("Componente Linear 1")
plt.ylabel("(n tem)")
plt.colorbar(label="Classe 'C'")
plt.show()


#%%
# Projetar os indivíduos desconhecidos no espaço das componentes discriminantes
# Selecionar as colunas corretas (as mesmas que foram usadas para treinar o LDA)
# Projetar os indivíduos desconhecidos no espaço LDA  # Substitua com a localização correta dos dados desconhecidos
# Projetando no espaço LDA (usando a transformação)
infarto_inconnu_transformee = lda.transform(individus_inconnu)

# Visualizando os indivíduos desconhecidos no espaço LDA
plt.figure(figsize=(8, 6))
plt.scatter(infarto_transformee[:, 0], np.zeros_like(infarto_transformee[:, 0]), c=infarto_connu_variable['C'], cmap='viridis', label='Indivíduos conhecidos')
plt.scatter(infarto_inconnu_transformee[:, 0], np.zeros_like(infarto_inconnu_transformee[:, 0]), c='red', marker='x', label='Indivíduos desconhecidos')
plt.title("Projetando indivíduos desconhecidos no espaço LDA")
plt.xlabel("Componente Linear 1")
plt.ylabel("(n existe)")
plt.colorbar(label="Classe 'C'")
plt.legend()
plt.show()