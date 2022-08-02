
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()
#print(bc)

x =scale(bc.data)
#print(x)
y =bc.target
#algorithm
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2)
model = KMeans(n_clusters=2,random_state=0)
model.fit(x_train)
predictions = model.predict(x_test)
labels = model.labels_
print(pd.crosstab(y_train,labels)) #cross tabulation of the data

print('labels',labels)
print('predictions',predictions)
print('actual_values',y_test)
print('accuracyscore',accuracy_score(y_test,predictions))
