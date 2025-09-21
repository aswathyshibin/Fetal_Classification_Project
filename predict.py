import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths
model_path = r"D:\MAITEXA TECHNOLOGIES\PROJECTS\test_2\Fetal_Classification_Project\models\fetal_classifier.keras"
labels_path = r"D:\MAITEXA TECHNOLOGIES\PROJECTS\test_2\Fetal_Classification_Project\data\labels.csv"

# Load model
print("🔄 Loading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Re-create label encoder from CSV
labels_df = pd.read_csv(labels_path)
encoder = LabelEncoder()
encoder.fit(labels_df['label'])

# Image size (must match training)
IMG_SIZE = 128

def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ File not found:", image_path)
        return

    # Preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = encoder.inverse_transform([class_idx])[0]

    print(f"✅ Prediction: {class_label} (score={prediction[0][class_idx]:.4f})")

# Run
if __name__ == "__main__":
    img_path = input("Enter the path of the image to test: ")
    predict_image(img_path)
