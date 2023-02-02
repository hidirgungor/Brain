import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Veri kümesini yükle
iris = pd.read_csv("iris.csv")

# Veri kümesini eğitim ve test verilerine böl
X = iris.drop("Species", axis=1)
y = iris["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# KNN modelini eğit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Modeli test verileri ile sına
y_pred = knn.predict(X_test)

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk oranı:", accuracy)

#Bu örnekte, iris veri kümesindeki verilerin %80'i eğitim verileri olarak kullanılır ve %20'si test verileri olarak kullanılır. KNN (K en yakın komşu) yöntemi ile sınıflandırma yapılır ve sonuçların doğruluk oranı hesaplanır.