#%% Import
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.signal.windows as scw
#%% Get dataset
iris = datasets.load_iris()

# On peut ensuite se ramener à une DataFrame
iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['Group'] = iris.target

[lines,columns] = iris_df.shape

iris_df.info()

#%% Réaliser une analyse univariée des variables 

plt.close('all')
# Boxplot de chaque variable en tenant compte des espèces ainsi que la moyenne
# et l variance de chaque variable en tenant compte des espèces
iris_df.boxplot(by = 'Group')
print(iris_df.groupby('Group').mean())
print(iris_df.groupby('Group').var())

#%% Réaliser une analyse bivariée (ADL)


# Linear Discriminant Analysis (LDA) is a statistical technique used for 
# classification and dimensionality reduction. It's commonly used to find a 
# linear combination of features that characterizes or separates two or more 
# classes of objects or events.

pd.plotting.scatter_matrix(iris_df.iloc[:, 0:4], c=iris_df['Group'])
adl = LinearDiscriminantAnalysis()
iris_transformee = adl.fit_transform(iris_df.iloc[:,0:4],iris_df['Group'])

#%% Représentation des iris dans le plan des 2 vecteurs discriminants

# Après avoir ajusté le modèle LDA, vous avez déjà obtenu les coordonnées 
# transformées des échantillons dans le plan des deux vecteurs discriminants. 
# Vous pouvez utiliser ces coordonnées pour visualiser les iris dans ce plan.
# Plotting iris data in the plane of the two discriminant vectors
plt.figure()
colors = ['navy', 'yellow', 'purple']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(iris_transformee[iris_df['Group'] == i][:, 0],
                iris_transformee[iris_df['Group'] == i][:, 1],
                color=color,
                alpha=.8,
                lw=lw,
                label=target_name)

    
plt.title('LDA of IRIS dataset')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('First Discriminant Vector')
plt.ylabel('Second Discriminant Vector')
plt.show()


# 2) Le pourcentage de variance expliquée par les 2 axes principaux
# Graphique des variances expliquées
plt.figure()

# This line creates a bar chart where each bar represents the variance explained 
# by each principal component. np.arange(1, p+1) generates an array of integers 
# from 1 to p, where p is the number of principal components. 
# acp.explained_variance_ratio_ contains the explained variance ratio of each 
# principal component.
plt.bar(range(1,len(adl.explained_variance_ratio_)+1),
        adl.explained_variance_ratio_)
# This line plots the cumulative sum of explained variances as a line graph. 
# np.cumsum() computes the cumulative sum of the explained variance ratio. 
# This line helps visualize how much variance is explained cumulatively as more 
# principal components are considered.
plt.plot(range(1,len(adl.explained_variance_ratio_)+1),
         np.cumsum(adl.explained_variance_ratio_),color='red')
plt.title('Pareto de la variance expliquée')
plt.xlabel('Nombres de facteurs  ')
plt.ylabel('capacité discriminant %')
plt.show()



#%%  Calcul de la corrélation entre chaque composante discriminante et les variables 
# de départ et tracé du cercle des corrélations.

# Calculate correlations between discriminant components and original variables
correlations = np.zeros((4, 2))

for i in range(4):
    for j in range(2):
        correlations[i, j] = np.corrcoef(iris_df.iloc[:, i], iris_transformee[:, j])[0, 1]

# Plotting correlation circle
plt.figure()
for i, (x, y) in enumerate(zip(correlations[:, 0], correlations[:, 1])):
    plt.plot([0, x], [0, y], 'k--')
    plt.text(x, y, iris.feature_names[i], fontsize='9', va='bottom')

# Add circles with different radius
circle = plt.Circle((0, 0), radius=1, color='b', fill=False)
plt.gca().add_artist(circle)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('First Discriminant Component')
plt.ylabel('Second Discriminant Component')
plt.title('Correlation Circle')
plt.grid()
plt.show()


#%% Calcul des centres de gravité de chaque classe dans l'espace de départ et 
# ajout des coordonnées des centres de gravité dans le plan des deux vecteurs 
# discriminants

# Correcting the calculation of centroids in the original space
centroids = iris_df.groupby('Group').mean().values

# Transforming centroids to the discriminant space
centroids_lda = adl.transform(centroids)

# Plotting iris data and centroids in the plane of the two discriminant vectors
plt.figure()
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
