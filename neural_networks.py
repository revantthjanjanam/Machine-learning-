from PIL import Image
import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()