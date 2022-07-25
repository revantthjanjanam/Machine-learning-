import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#features/labels
x = boston.data
y = boston.target
#data
'''
print(x)
print(y)
print(x.shape)
print(y.shape)
'''
#linear regression algorithm
l_reg = linear_model.LinearRegression()
#cheaking appropriate data
plt.scatter(x.T[5],y)#x.T[] gives the transpose of 
plt.show()

x_train,x_test,y_train,y_test = train_test_split (x,y, test_size=0.2)

#train
model = l_reg.fit(x_train, y_train)
predictions = model.predict(x_test)
print('predictions: ',predictions)
print('r^2 value:   ',l_reg.score(x,y))
print('coef:    ',l_reg.coef_)
print('intercept:   ',l_reg.intercept_)