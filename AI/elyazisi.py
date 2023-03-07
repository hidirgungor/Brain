import cv2
import numpy as np
from keras.models import load_model

# Eğitilmiş modeli yükle
model = load_model('model.h5')

# Test resmini yükle ve ön işleme tabi tut
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)
img = img.reshape(1, 28, 28, 1)

# Tahmin yap
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

# Sonucu ekrana yazdır
print('Predicted digit:', predicted_digit)
