#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

#%% Get data from excel
plt.close('all')
# On charge les données
notes = pd.read_excel("Notes.xlsx",sheet_name=0,header=0,index_col=0)

[lines,columns] = notes.shape #n=notes.shape[0]

notes.info()


#%% Basic analysis

## Get overall panorama
print(notes.describe())

## Get mean
print(notes.mean())

## Get variance
print(notes.var())

# notes.boxplot()

#%% Analyses bivariées des variables quantitatives

# math in function of physics or vice versa
# plt.scatter(notes["MATH"], notes["PHYS"])

# for i in range(lines):
#  plt.annotate(notes.index[i], (notes.iloc[i, 0], notes.iloc[i, 1]))
 
# plt.xlabel('MATH')
# plt.ylabel('PHYS')

#%% Same analysis but for all the variable pairs

# pd.plotting.scatter_matrix(notes)

correlation = pd.DataFrame.corr(notes, method='pearson')
plt.figure()

# sns.heatmap(correlation, vmin=-1, vmax=1, cmap='coolwarm', annot=True)

# MATRIX OF COVARIANCE
covariance = pd.DataFrame.cov(notes)
print(covariance)

corr = np.corrcoef(notes, rowvar=False)
eig_corr = np.linalg.eig(corr)
#%% Réaliser l’ACP - Analyse en composantes principales
#pca: Principal component analysis 

acp = PCA()

# Réaliser une ACP sur les données et retourne dans Xacp les notes_acponnées du
# nuage de points-individus dans le nouvel espace (espace des axes principaux 
# qui sont les vecteurs propres de la matrice de variance-covariance des données)
notes_acp = acp.fit_transform(notes)

# Graphique des variances expliquées
plt.figure()

# This line creates a bar chart where each bar represents the variance explained 
# by each principal component. np.arange(1, p+1) generates an array of integers 
# from 1 to p, where p is the number of principal components. 
# acp.explained_variance_ratio_ contains the explained variance ratio of each 
# principal component.
plt.bar(np.arange(1, columns+1), acp.explained_variance_ratio_)
# This line plots the cumulative sum of explained variances as a line graph. 
# np.cumsum() computes the cumulative sum of the explained variance ratio. 
# This line helps visualize how much variance is explained cumulatively as more 
# principal components are considered.
plt.plot(np.arange(1, columns+1), np.cumsum(acp.explained_variance_ratio_))
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de facteurs")
plt.show()

# Explained variance,in the context of PCA, refers to the amount of variance in
# the original data that is explained by each principal component.

# Principal components are new variables that are constructed as linear 
# combinations of the original variables. The first principal component accounts 
# for the most variance in the data, the second principal component accounts for 
# the second most variance, and so on.

# When we say "explained variance" for a principal component, we mean the 
# proportion of the total variance in the data that is explained by that 
# particular principal component. In other words, it quantifies how much of the 
# information in the dataset is captured by each principal component.

# Explained variance is typically expressed as a ratio or percentage. In the 
# context of PCA, it's computed by dividing the variance of each principal 
# component by the total variance of the original data.

#%% Resultat de l'ACP

# 1) Les individus projetés dans l’espace des 2 axes principaux
# Les individus ne sont pas correlés aux composantes
plt.scatter(notes_acp[:, 0], notes_acp[:, 1])
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')


# 2) Le pourcentage de variance expliquée par les 2 axes principaux
# # Graphique des variances expliquées
# plt.figure()

# # This line creates a bar chart where each bar represents the variance explained 
# # by each principal component. np.arange(1, p+1) generates an array of integers 
# # from 1 to p, where p is the number of principal components. 
# # acp.explained_variance_ratio_ contains the explained variance ratio of each 
# # principal component.
# plt.bar(np.arange(1, columns+1), acp.explained_variance_ratio_)
# # This line plots the cumulative sum of explained variances as a line graph. 
# # np.cumsum() computes the cumulative sum of the explained variance ratio. 
# # This line helps visualize how much variance is explained cumulatively as more 
# # principal components are considered.
# plt.plot(np.arange(1, columns+1), np.cumsum(acp.explained_variance_ratio_))
# plt.ylabel("Variance expliquée en ratio et cumul")
# plt.xlabel("Nombre de facteurs")
# plt.show()

# 3) L’interprétation des axes : en regardant la corrélation des 2 composantes 
# principales avec les variables de départ. !!! LES VARIABLES SONT CORRELES AUX AXIS

# On va ici calculer la corrélation entre les facteurs et les variables de départ
XArray = pd.DataFrame.to_numpy(notes)
corvar = np.zeros((4, 2))
corvar[0, 0] = np.corrcoef(notes_acp[:, 0], XArray[:, 0])[0, 1]
corvar[1, 0] = np.corrcoef(notes_acp[:, 0], XArray[:, 1])[0, 1]
corvar[2, 0] = np.corrcoef(notes_acp[:, 0], XArray[:, 2])[0, 1]
corvar[3, 0] = np.corrcoef(notes_acp[:, 0], XArray[:, 3])[0, 1]
corvar[0, 1] = np.corrcoef(notes_acp[:, 1], XArray[:, 0])[0, 1]
corvar[1, 1] = np.corrcoef(notes_acp[:, 1], XArray[:, 1])[0, 1]
corvar[2, 1] = np.corrcoef(notes_acp[:, 1], XArray[:, 2])[0, 1]
corvar[3, 1] = np.corrcoef(notes_acp[:, 1], XArray[:, 3])[0, 1]

# afficher la matrice des corrélations 2 premiers facteurs avec les variables de départ
print(corvar)
print(pd.DataFrame({'id': notes.columns, 'COR_1': corvar[:, 0], 'COR_2': corvar[:, 1]}))

# Cercle des corrélations
fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
plt.scatter(corvar[:, 0], corvar[:, 1])

# affichage des étiquettes (noms des variables)
for j in range(columns):
 plt.annotate(notes.columns[j], (corvar[j, 0], corvar[j, 1]))
 
# On ajoute les axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# On ajoute un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)
plt.show()