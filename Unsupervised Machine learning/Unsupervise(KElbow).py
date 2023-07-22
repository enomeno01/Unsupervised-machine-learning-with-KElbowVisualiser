# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:51:13 2023

@author: enesd
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

data = pd.read_csv("C:/Users/enesd/Downloads/Mall_customers.csv")
dt=data.copy()

dt=dt.drop(columns="CustomerID",axis=1)

X=dt.iloc[:,1:3]


kmodel=KMeans(random_state=(0))

grafik=KElbowVisualizer(kmodel, k=(1,20))
grafik.fit(X)
grafik.poof()



kmodel=KMeans(n_clusters=4,random_state=(0))


kfit=kmodel.fit(X)
kumeler=kfit.labels_
merkezler=kfit.cluster_centers_

figure,axis=plt.subplots(1,2)
axis[0].scatter(X.iloc[:,0],X.iloc[:,1],color="black")
axis[1].scatter(X.iloc[:,0],X.iloc[:,1],c=kumeler,cmap="winter")
axis[1].scatter(merkezler[:,0],merkezler[:,1],c="red",s=200)
