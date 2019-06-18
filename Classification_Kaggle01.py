import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
x_train=data.iloc[:,1:55].values
y_train=data.iloc[:,-1].values

data_test=pd.read_csv('test.csv')
x_test=data_test.iloc[:,1:].values
y_test=data_test.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=10)
x_train=lda.fit_transform(x_train,y_train)
x_test=lda.transform(x_test)

from sklearn.ensemble import RandomForestClassifier as RFC
model=RFC(n_estimators=400,criterion='entropy',random_state=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
