from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

x = iris.data
y = iris.target

classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']

#svm algorithm
x_train,x_test,y_train,y_test = train_test_split (x,y, test_size=0.2)
svm = svm.SVC()
svm.fit(x_train,y_train)
predictions = svm.predict(x_test)

#measuring accuracy
accuracy = metrics.accuracy_score(y_test,predictions)

print('Predictions  :',predictions)
print("actual value :",y_test)
print('accuracy score   :',accuracy)

for i in range(len(predictions)):
    print(classes[predictions[i]])
