from operator import irshift
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()

x = iris.data
y = iris.target
print (x,y)

x_train,x_test,y_train,y_test = train_test_split (x,y, test_size=0.2)
print("\nxt", x_train,"\nxtt",x_test,"\nytn",y_train,"\nytt",y_test)

from sklearn.preprocessing import LabelEncoder
dir(LabelEncoder)