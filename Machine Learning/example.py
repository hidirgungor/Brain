# IRIS Data Library
# Bu örnek veri setini eğitim ve test verileri olarak böler, KNN modelini oluşturur ve eğitir, sonuç olarak modelin test verileri üzerindeki doğruluğunu yazdırır.
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Iris veri kümesini yükle
iris = datasets.load_iris()

# veri ve hedef değişkenleri ayır
X = iris.data
y = iris.target

# veri setini eğitim ve test verileri olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# modelin test verileri üzerindeki doğruluğunu ölç
accuracy = knn.score(X_test, y_test)
print("Accuracy: ", accuracy)
