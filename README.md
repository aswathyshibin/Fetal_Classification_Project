# Fetal Image Classification – Normal vs Abnormal

This project is a deep learning solution for classifying **fetal ultrasound images** as either **Normal** or **Abnormal**.  
It is built with **Python, TensorFlow/Keras, NumPy, Pandas, and OpenCV**.  

The goal is to support medical imaging research and demonstrate how Convolutional Neural Networks (CNNs) can be applied to healthcare problems.

---

##  Project Overview

- **Data**: Ultrasound images + a `labels.csv` file that maps each image to `normal` or `abnormal`.  
- **Model**: A CNN trained to recognize features from ultrasound scans.  
- **Training**: Images are preprocessed (grayscale, resized, normalized), then fed into the CNN.  
- **Prediction**: Once trained, the model can classify new unseen images in just a few milliseconds.  

---

##  Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/Fetal-Image-Classification.git
cd Fetal-Image-Classification

Create a Virtual Environment

python -m venv venv

Install Dependencies

pip install -r requirements.txt

Dataset

Input: Ultrasound fetal images (.png)

Labels: labels.csv with two columns:

filename → image name

label → normal or abnormal

⚠️ The dataset is not included here due to size limitations.

Tech Stack

Python 3.x

TensorFlow / Keras for deep learning

NumPy, Pandas for data handling

OpenCV for image preprocessing

Matplotlib for visualization

Developed by ASWATHY N M
 Contact: aswathyshibinm@gmail.com
