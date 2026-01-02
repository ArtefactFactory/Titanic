# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 08:02:07 2025

@author: Vincent
"""

# Import des librairies et des données
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv/titanic.csv")

#Visualisation du jeu de données avant nettoyage
df.info()
df.head(10)
print(df.isna().sum())

# Nettoyage du jeu de données
df.dropna(subset=["Age", "Embarked"], inplace=True) # on supprime les NaN dans les colonnes "Age" et "Embarked"
df["SibSp"] = df["SibSp"] + df["Parch"]
df.rename(columns = {"SibSp":"Relatives"}, inplace = True)
df.drop(["Name","Cabin", "Parch"], axis=1, inplace=True)

# Visualisation du jeu de données après nettoyage
print(df.isna().sum())
df.head()
df.describe()

# 2. Qui étaient les passagers ?
# création d'une boîte à moustaches pour analyser la colonne "Âge"
plt.figure()
df.boxplot(column= ["Age"])

# création de 3 sous-dataframes pour les hommes, femmes et enfants
df_men = df[(df["Sex"]=="male") & (df["Age"]>=18)]
df_women = df[(df["Sex"]=="female") & (df["Age"]>=18)]
df_children = df[df["Age"]<18]

# visualisation graphique de plusieurs données
colors = ["purple","orange","green"]
groupes = ["Hommes", "Femmes", "Enfants"]
plt.style.use("bmh")

# bar plot montrant la répartition hommes/femmes/enfants
plt.figure()
plt.bar("Hommes", df_men["Sex"].value_counts(), label = groupes[0], color = colors[0])
plt.bar("Femmes", df_women["Sex"].value_counts(), label = groupes[1], color = colors[1])
plt.bar("Enfants", df_children["Sex"].value_counts(), label = groupes[2], color = colors[2])
plt.legend(groupes)

# histogramme des âges
plt.figure()
df["Age"].hist(bins=20)
plt.xlabel("Age")
plt.ylabel("Fréquence")

# calcul des tendances centrales pour la colonne "Âge"
mean_age = df["Age"].mean()
median_age = df["Age"].median()
mode_age = df["Age"].mode()
print(f"Âge moyen = {np.round(mean_age, decimals=1)} ans")
print(f"Âge médian = {median_age}")
print(f"Mode = {mode_age} ans")

# histogramme permettant d'observer comment sont répartis les hommes/femmes/enfants par tranches d'âge
plt.figure()
plt.hist( [df_men["Age"],df_women["Age"], df_children["Age"]], bins=[0,10,20,30,40,50,60,70,80] , color = colors )
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.legend(["Hommes", "Femmes", "Enfants"])

# histogramme permettant d'observer comment sont répartis les hommes/femmes/enfants par classe
plt.figure()
plt.hist( [df_men["Pclass"],df_women["Pclass"], df_children["Pclass"]], bins = 3, color = colors )
plt.xticks([1,2,3])
plt.xlabel("Classe")
plt.ylabel("Fréquence")
plt.legend(["Hommes", "Femmes", "Enfants"])

# histogramme permettant d'observer s'il y a un lien entre la ville d'embarquement et la classe dans le bateau
plt.figure()
plt.hist( [df[df["Pclass"]==3]["Embarked"], df[df["Pclass"]==2]["Embarked"], df[df["Pclass"]==1]["Embarked"]], bins = 3 )
plt.xticks(ticks=pd.unique(df["Embarked"]))
plt.xlabel("Ville d'embarquement")
plt.ylabel("Fréquence")
plt.margins(x = 0.1)
plt.legend(["3ème", "2ème", "1ère"])

# histogramme permettant d'observer s'il y a un lien entre la ville d'embarquement et la classe dans le bateau
plt.figure()
plt.hist( [df[df["Pclass"]==3]["Embarked"], df[df["Pclass"]==2]["Embarked"], df[df["Pclass"]==1]["Embarked"]], bins = 3 )
plt.xticks(ticks=pd.unique(df["Embarked"]))
plt.xlabel("Ville d'embarquement")
plt.ylabel("Fréquence")
plt.margins(x = 0.1)
plt.legend(["3ème", "2ème", "1ère"])

# pie chart permettant d'observer si les passagers voyagaient plutôt seul ou avec des proches
plt.figure()
plt.pie(df["Relatives"].value_counts(), autopct = "%1.1f%%")
plt.legend(["0","1","2","3","4","5","6","7"])

#3. Facteur de survie
# calcul de moyenne dans des regroupements
df.groupby(["Sex"]).mean()
df.groupby(["Sex"]).median()
df.groupby(["Sex", "Pclass"]).mean(numeric_only=True)

# on observe la répartition hommes/femmes/enfants parmi les morts/survivants
plt.figure()
plt.hist([df_men["Survived"], df_women["Survived"], df_children["Survived"]], bins = 2)
plt.xticks(ticks=[0,1], labels=["Morts", "Vivants"])
plt.margins(x=0.1)
plt.legend(["Hommes", "Femmes", "Enfants"])

# on observe la répartition 3ème/2ème/1ère classes parmi les morts/survivants
plt.figure()
plt.hist([df[df["Pclass"]==3]["Survived"], df[df["Pclass"]==2]["Survived"], df[df["Pclass"]==1]["Survived"]], bins = 2)
plt.xticks(ticks=[0,1], labels=["Morts", "Vivants"])
plt.margins(x=0.1)
plt.legend(["3ème", "2ème", "1ère"])

