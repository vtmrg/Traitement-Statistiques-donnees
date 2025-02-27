import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import pylab
import pandas as pd
import seaborn as sns
import statistics
import statsmodels.api as smi
#Import data from csv (offline)
tips = pd.read_csv("tips/tips.csv")



#Get header and initial lines
print(tips.head())

#Get quick description of the data (columns, lines, data types, etc)
tips.info()

# Get a global panorama of the quantitative fields of the dataset
print(tips.describe())

# Get a global panorama of one of the quantitative fields
# Syntax can also be tips["total_bill"].describe()
print(print(tips.total_bill.describe()))

# The individual values can be obtained by the specific methods below:
# tips.total_bill.mean()
# tips.total_bill.std() 
# tips.total_bill.min()
# tips.total_bill.max() 
# tips.total_bill.median()
# tips.total_bill.quantile([0.25, 0.5, 0.75])
# tips.total_bill.var()

# !!! QUANTITATIVE DATA CAN BE DYSPLAYED BY: Histograms or BoxPlots !!!
# HISTOGRAMS:
    
#tips.plot.hist(bins = 20) #bins -> number of intervals of subdivision
#tips.total_bill.plot.hist(bins = 20) #Non Normalized
#tips.total_bill.plot.hist(bins = 20, density = True) # Normalized
##sns.distplot(tips.total_bill) # Normalized with an approx of the density func

# BOXPLOTS:
    
#tips.boxplot()
#tips.boxplot(column = "total_bill")
##sns.boxplot(y = "total_bill", data = tips)

# VIOLIN (mixture of both, each side is an estimation of the density)

#sns.violinplot(y = "total_bill", data = tips, color = "skyblue")

#%% Qualitative data analysis 

# Ces variables sont utilisées quand le caractère étudié n’est pas
# quantifiable.
# • Une variable catégorielle nominale décrit un nom, une
# étiquette ou une catégorie sans ordre naturel ; par exemple le
# sexe, ou le type de pathologie.
# • Une variable catégorielle ordinale est une variable dont les
# valeurs sont définies par une relation d’ordre entre les
# catégories possibles; par exemple le niveau de performance
# d’un athlète: « excellent » - « très bon » - « bon » …
# • Attention: il arrive que dans un jeu de donnée certaines
# variables catégorielles apparaissent sous la forme de nombre
# (les variables ont été codées). Il est important d’avoir la table
# de codage et surtout il est important d’indiquer au logiciel
# utilisé pour l’analyse que la variable correspond à une variable
# catégorielle et pas une variable numérique.


# Ask for the categories of a specific qualitative column
#print(tips.sex.unique())

# Get number in each category
#print(pd.crosstab(tips.sex, "freq"))

# Get the frequency of each category, which corresponds to the normalized 
#number of occurances
print(pd.crosstab(tips.sex, "freq", normalize = True))

# !!! QUALITATIVE DATA CAN BE DYSPLAYED BY: Bar diagrams or Pizza diagrams !!!
# BAR DIAGRAM:
    
# t = pd.crosstab(tips.sex, "freq")
# t = pd.crosstab(tips.sex, "freq", normalize= True)
# t.plot.bar()
##sns.countplot(x = "sex", data = tips)

# PIZZA DIAGRAM:
# t = pd.crosstab(tips.sex, "freq")
# t.plot.pie(subplots=True, figsize = (3, 3))

#%% Simultaneous analysis of quantitative data

# Create a scatterplot of total_bill vs. tip
#tips.plot.scatter("total_bill", "tip", color = "green")

# Get the coefficient of correlation between the data
corr = tips.total_bill.corr(tips.tip)
print(corr)

# Create a nice jointplot, with the histograms of each category, their density
#respectively, a linear regression,  and a "zone de confiance"
#sns.jointplot(x = "total_bill", y = "tip", data = tips, kind = "reg")

#%% Simultaneous analysis of qualitative data

# Get the contingency table between two categories
print(pd.crosstab(tips.sex, tips.smoker))

# Get a relational bar graph of the absolute values "efectifs"
# t = pd.crosstab(tips.sex, tips.smoker)
# t.plot.bar()

# Get a relational bar graph of the frequencies
# t = pd.crosstab(tips.sex, tips.smoker, normalize = True)
# t.plot.bar()

# Superpose the columns of the frequency bar graph
# t = pd.crosstab(tips.sex, tips.smoker, normalize = "index")
# t.plot.bar(stacked = True)

# Get Pizza charts of the 
# t = pd.crosstab(tips.sex, tips.smoker)
# t.plot.pie(subplots = True, figsize = (12, 6))
#%% Simultaneous analysis of quantitative data and a qualitative one

# Get the mean of the quantitative data (total_bill, tip and size) as function 
#of the sex
# We need to filter the used columns in order for the mean to work with the 
#groupby
t = tips.filter(items= ["sex", "total_bill","tip", "size"]).groupby("sex").mean()
print(t)

# Another solution is: 
   # Map string values to numeric values
   # tips['sex'] = tips['sex'].map({'Male': 0, 'Female': 1})
   
# In depth analysis of the total_bill grouped by sex
t = tips.groupby("sex")["total_bill"].agg([np.mean, np.std, np.median, np.min, np.max])
print(t)

# Graphical analysis of the histograms of the total_bill as function of the sex
sns.kdeplot(data = tips, x="total_bill", hue = "sex", multiple = "stack")

# Graph of the average and the confidence interval
sns.catplot(x = "sex", y = "total_bill", data = tips, kind = "point", join = False)

# Mean analysis with the boxplots
sns.catplot(x = "sex", y = "total_bill", data = tips, kind = "box")

# Violin analysis 
sns.catplot(x = "sex", y = "total_bill", data = tips, kind = "violin")
