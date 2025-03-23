#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:56:57 2025

@author: vitoriamariana
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#%%
# On charge les données
X = pd.read_excel("/Users/vitoriamariana/Desktop/Faculdade/Polytech/S8-2025/Traitement-Statistiques-donnees/TP02/Criminalite.xlsx",sheet_name=0,header=0,index_col=0)
print(X.shape) #ou [l,c]=X.shape ou l=X.shape[0]
print(X.info())

#%%
#%% Basic analysis

## Get overall panorama
print(X.describe())

## Get mean
print(X.mean())

## Get variance
print(X.var())

X.boxplot()


#%%
#2.3. Réaliser une analyse bivariée des variables quantitatives 
Cor=pd.plotting.scatter_matrix(X)
plt.show()

#%% Matrice en utilisant la heatmap de seaborn
correlation=X.corr()
print(correlation)
sns.heatmap(correlation,vmin=-1,vmax=1,cmap='coolwarm', annot=True)
plt.show()
#%%
#2.4. Réaliser une ACP sur les données centrées et sur les données centrées-réduites.
acp = PCA()
#Entender a diferenca entre fit e transform
#Fit the model with X and apply the dimensionality reduction on X.
Xacp = acp.fit_transform(X)
print(Xacp)
