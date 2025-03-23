#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 08:28:39 2025

@author: vitoriamariana
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#%%
iris = datasets.load_iris()

#On peut ensuite se ramener à une DataFrame
Xdf = pd.DataFrame(iris.data,columns=iris.feature_names)
Xdf['Group']=iris.target
#%%
Xdf.boxplot(by="Group")
Xdf.groupby('Group').mean()
#%%
#3. Réaliser une analyse bivariée des variables quantitatives 
Xvariable=Xdf.iloc[:,0:-1]
Cor=pd.plotting.scatter_matrix(Xvariable,c=Xdf['Group'])
plt.show()

#%% Matrice en utilisant la heatmap de seaborn
correlation=Xdf.corr()
print(correlation)
sns.heatmap(correlation,vmin=-1,vmax=1,cmap='coolwarm', annot=True)
plt.show()
#%%
#4/ Réaliser l’analyse multivariée (approche statistiques descriptives). 
#1.- Vous représenterez les iris dans le plan des 2 vecteurs discriminants 
#2.
lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(Xdf.iloc[:,0:4],Xdf['Group'])

#%%
#Représentation des iris dans le plan des 2 vecteurs discriminants

plt.figure()
plt.plot(coord_lda, 'o')
plt.xlabel("composant 1")
plt.ylabel("composant 2")


#%%
#Vous calculerez la corrélation entre chaque composante discriminante et les variables de départ pour
#tracer le cercle des corrélations qui permet d’interpréter les nouveaux axes
correlations = np.zeros((4, 2)) #por enquanto nulo

#entre sepal length et composant 1
correlations[0,0]=np.corrcoef(Xdf.iloc[:,0],coord_lda[:,0])[0, 1]
#entre sepal length et composant 2
correlations[0,1]=np.corrcoef(Xdf.iloc[:,0],coord_lda[:,1])[0, 1]


#entre sepal width et composant 1
correlations[1,0]=np.corrcoef(Xdf.iloc[:,1],coord_lda[:,0])[0, 1]
#entre sepal length et composant 2
correlations[1,1]=np.corrcoef(Xdf.iloc[:,1],coord_lda[:,1])[0, 1]

#entre petal length et composant 1
correlations[2,0]=np.corrcoef(Xdf.iloc[:,2],coord_lda[:,0])[0, 1]
#entre petal length et composant 2
correlations[2,1]=np.corrcoef(Xdf.iloc[:,2],coord_lda[:,1])[0, 1]

#entre petal width et composant 1
correlations[3,0]=np.corrcoef(Xdf.iloc[:,3],coord_lda[:,0])[0, 1]
#entre petal width et composant 2
correlations[3,1]=np.corrcoef(Xdf.iloc[:,3],coord_lda[:,1])[0, 1]


#%%
plt.figure(figsize=(10, 8))  
circle = plt.Circle((0, 0), radius=1, color='b', fill=False)
plt.gca().add_artist(circle)
for i, (x, y) in enumerate(zip(correlations[:, 0], correlations[:, 1])):
    plt.plot([0, x], [0, y], 'k--')
    plt.text(x, y, iris.feature_names[i], fontsize='6', va='bottom')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('First Discriminant Component')
plt.ylabel('Second Discriminant Component')
plt.title('Correlation Circle')
plt.grid()
plt.show()

#Depois perguntar pro chat pq esse codigo é diferente de plt.plot(correlations)(na minha cabeca é a mesma coisa)

#%% Calcul des centres de gravité de chaque classe dans l'espace de départ et 
# ajout des coordonnées des centres de gravité dans le plan des deux vecteurs 
# discriminants

centro_de_gravidade = Xdf.groupby('Group').mean().values
adl = LinearDiscriminantAnalysis()
# Transforming centroids to the discriminant space
#fazer a adl
#iris_transformee = adl.fit_transform(iris_df.iloc[:,0:4],iris_df['Group'])
cdg_lda = lda.fit(X,y).transform(centro_de_gravidade)


#%%




# Plotting iris data and centroids in the plane of the two discriminant vectors
plt.figure(figsize=(10, 8))  # Adjusting figure size
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(iris_transformee[iris_df['Group'] == i][:, 0],
                iris_transformee[iris_df['Group'] == i][:, 1],
                color=color,
                alpha=.8,
                lw=lw,
                label=target_name)

plt.scatter(centroids_lda[:, 0],
            centroids_lda[:, 1],
            color='k',
            marker='x',
            s=100,
            label='Centroids')

plt.title('LDA of IRIS dataset with Centroids')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('First Discriminant Vector')
plt.ylabel('Second Discriminant Vector')
plt.show()
