import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
x = data[[
    'buying',
    'maint',
    'safety'
]].values
y=data['class']

le = LabelEncoder()
#converting x data
for i in range(len(x[0])):
    x[:,i] = le.fit_transform(x[:,i])

#converting y data
label_mapping={
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y = y.map(label_mapping)
y =np.array(y)

#knn algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

knn.fit(x_train,y_train)

predictions = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test,predictions)

print('predictions  :\n',predictions)
print('actual value :\n',y_test)
print("accuracy :",accuracy)
#testing random
test = 1043
print("actual value:",y[test])
print("Predicted value :", knn.predict(x)[test]) 