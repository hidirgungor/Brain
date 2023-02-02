import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Verileri okuma
df = pd.read_csv("spam.csv")

# Verileri hazırlama
X = df["email"]
y = df["label"]
le = LabelEncoder()
y = le.fit_transform(y)

# E-postaları sözlük şekline dönüştürme
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Verileri eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Sinir ağını oluşturma ve eğitme
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

# Test verilerini kullanarak performansı ölçme
y_pred = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


#Bu örnekte, veriler e-posta içeriği ve spam/ham etiketleri şeklinde okunmuş ve hazırlanmıştır. Daha sonra veriler sözlük şekline dönüştürülmüş ve eğitim ve test verilerine ayırılmıştır. Son olarak, bir sinir ağı oluşturulmuş ve eğitilmiştir. Eğitim verileri kullanılarak yapılan tahminlerin doğruluğu ise accuracy_score fonksiyonu ile ölçülmüştür.