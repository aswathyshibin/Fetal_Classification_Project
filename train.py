import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Paths
image_folder = r"D:\MAITEXA TECHNOLOGIES\PROJECTS\test_2\Fetal_Classification_Project\data\IMAGES"
labels_path = r"D:\MAITEXA TECHNOLOGIES\PROJECTS\test_2\Fetal_Classification_Project\data\labels.csv"
model_save_path = r"D:\MAITEXA TECHNOLOGIES\PROJECTS\test_2\Fetal_Classification_Project\models\fetal_classifier.keras"

# Load labels
labels_df = pd.read_csv(labels_path)

# Image size (resize to fixed size for CNN)
IMG_SIZE = 128

# Preprocess images
X = []
y = []

for idx, row in labels_df.iterrows():
    img_path = os.path.join(image_folder, row['filename'].replace(".npy", ".png"))  # Adjust extension if needed
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # ultrasound is grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # normalize
        X.append(img)
        y.append(row['label'])

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # normal=0, abnormal=1
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# CNN Model (improved with dropout to reduce overfitting)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)

# Save model in new Keras format
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"✅ Model saved at {model_save_path}")

# Plot training results
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

# --- Prediction Function ---
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = encoder.inverse_transform([class_idx])[0]

    print(f"✅ Prediction: {class_label} (score={prediction[0][class_idx]:.4f})")

# Example test
if __name__ == "__main__":
    img_path = input("Enter the path of the image to test: ")
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print("❌ File not found. Please check the path again.")
