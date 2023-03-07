import tensorflow as tf
from tensorflow import keras

# Veri kümesini yükle
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Veri kümesini normalize et
x_train = x_train / 255.0
x_test = x_test / 255.0

# Modeli tanımla
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Modeli derle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=10)

# Modeli değerlendir
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
