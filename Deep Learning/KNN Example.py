from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Modeli eğit
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Modeli test et
test = [[3, 5, 4, 2]]
print(model.predict(test))
