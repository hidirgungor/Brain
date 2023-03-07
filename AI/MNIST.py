from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

# Veri setini yükleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Verileri ön işleme
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeli oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Modeli derleme ve eğitim
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
