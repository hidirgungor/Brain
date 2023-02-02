import numpy as np
import matplotlib.pyplot as plt

# Verileri oluşturma
np.random.seed(0)
X = np.random.normal(0, 1, (100, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Ağırlıkları ve başlangıç bias değerlerini oluşturma
weights = np.random.normal(0, 1, (2, 1))
bias = np.zeros((1, 1))

# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Üstel eğitim fonksiyonu
def logistic_regression(X, y, weights, bias, num_iterations, learning_rate):
    m = X.shape[0]
    costs = []
    for i in range(num_iterations):
        z = np.dot(X, weights) + bias
        a = sigmoid(z)
        cost = - (1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        dz = a - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        costs.append(cost)
    return weights, bias, costs

# Eğitim işlemini yapma
weights, bias, costs = logistic_regression(X, y, weights, bias, num_iterations=1000, learning_rate=0.01)

# Eğitilen ağırlıkları ve bias değerlerini görme
print("Ağırlıklar:", weights)
print("Bias Değeri:", bias)

# Maliyet değişim grafiği
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


#Bu örnekte, rastgele olarak oluşturulan verilere dayanarak, logistic regresyon algoritması ile bir sinir ağının eğitilmesi gösterilmiştir. Eğitim süreci boyunca maliyet değeri grafiği de görüntülenmiştir.