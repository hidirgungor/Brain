import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

# Veri oluşturma
np.random.seed(0)
x = np.linspace(0, 10, 500)
y = np.sin(x) + np.random.normal(0, 0.1, 500)

# Veri düzenleme
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

# Model oluşturma
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Model eğitme
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=32)

# Model tahmin etme
x_test = np.linspace(0, 10, 100)
X_test = x_test.reshape(-1, 1)
y_pred = model.predict(X_test)

# Tahmin sonuçlarını görselleştirme
plt.plot(x_test, y_pred, 'r', label='Prediction')
plt.scatter(X, Y, label='Data')
plt.legend()
plt.show()
