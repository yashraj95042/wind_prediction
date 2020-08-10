#predicting the wind generation
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('weather_data_GER_2016.csv')

X = dataset.iloc[:, 4:7].values
y = dataset.iloc[:, 9].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0) 
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Z = regressor.predict(X_test)
print(regressor.intercept_)
print(regressor.coef_)
print('Mean Squared Error:',mean_squared_error(Z,y_test))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(Z,y_test)))
print('r_2 Statistic : %.2f' % r2_score(Z,y_test))

#print(Z.round(2))
#print (X_train)
#print(y_train)
#Z = Z.round(2)
#m=np.array([X_test])
#n=np.array([y_test])
#print(m.shape)
#print(n.shape)
df = pd.DataFrame({'Actual': Z, 'Predicted': y_test})
print(df)
plt.scatter(Z,y_test,color='red')
#plt.plot(X_test,y_test,color='blue')
plt.show()