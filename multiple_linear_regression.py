#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Encoing categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#проблема, которая может возникнут, заключается в том, что модель будет думать,
#что страна с цифрой 2 больше страны с цифрой 1 и 0, итд  
#чтобы решить эту проблему, мы сделаем три столбцы(из примера задачи) Германия Франция и Испания,
#каждый из которых будет содержать 1 в той строке, где есть именно эта страна и 0, где остальные страны 

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap 
X = X[:, 1:]



#если мы будем применять все выше указанное к зависимой переменной, то нам не надо будет разбивать 
#ее на несколько столбцов, потому что модель будет знать, что эта переменная - зависимая 
#и между ее значениями нет порядка


#Splitting dataset to training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#применяем multiple linear regressor к нашему тренировочному набору данных
regressor.fit(X_train, Y_train)

#Predicting the Test Set
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
#наполянем нашу таблицу единицами в первом ряду, чтобы у нас был "живой" b0, 
#таким образом b0x0 будет равно b0, так как весь наш первый ряд равен единицам, 
#что никак не изменяет нашу переменную b0
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X , axis = 1)
#x_opt содержит только те назависимые переменные, которые имеют большое влияние на 
#зависимую переменную
X_opt = X[:, [0,1,2,3,4,5]]

#ordinary list squares 
regressor_OLS = sm.OLS( endog = Y, exog = X_opt ).fit()

regressor_OLS.summary()
#-------------#
X_opt = X[:, [0,1,3,4,5]]

#ordinary list squares 
regressor_OLS = sm.OLS( endog = Y, exog = X_opt ).fit()

regressor_OLS.summary()

#-------------#
X_opt = X[:, [0,3,4,5]]

#ordinary list squares 
regressor_OLS = sm.OLS( endog = Y, exog = X_opt ).fit()

regressor_OLS.summary()

#-------------#
X_opt = X[:, [0,3,5]]

#ordinary list squares 
regressor_OLS = sm.OLS( endog = Y, exog = X_opt ).fit()

regressor_OLS.summary()

#-------------#
X_opt = X[:, [0,3]]

#ordinary list squares 
regressor_OLS = sm.OLS( endog = Y, exog = X_opt ).fit()

regressor_OLS.summary()





