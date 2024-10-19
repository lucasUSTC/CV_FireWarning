import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return np.array(images)

def extract_hsv_features(images):
    hsv_features = []
    for img in images:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        hsv_feature = [h_mean, s_mean]
        hsv_features.append(hsv_feature)
    return np.array(hsv_features)

# Function to predict the cluster for new images
def predict_cluster_for_new_images(new_images_folder, model_path):
    kmeans_model = joblib.load(model_path)
    new_images = load_images_from_folder(new_images_folder)

    # Check if new images are loaded
    if new_images.size == 0:
        raise ValueError("No images found in the new images folder.")

    # Extract HSV features for new images
    new_hsv_features = extract_hsv_features(new_images)

    # Predict clusters for new images
    predictions = kmeans_model.predict(new_hsv_features)

    # Map clusters to temperature ranges
    class_labels = ['260℃', '50℃', '150℃', '240℃', '100℃', '200℃', '25℃', '250℃']
    temperature_predictions = [class_labels[pred] for pred in predictions]

    return temperature_predictions


warning_list =[ ['260℃'] ]

warning = cv2.imread('/content/sample_data/fire.png')

# repalce your file path
new_images_folder = '/content/sample_data/test'
model_path = '/content/sample_data/kmeans_model.joblib'
predictions = predict_cluster_for_new_images(new_images_folder, model_path)
print("Predicted clusters for new images:", predictions)

if predictions in warning_list:
    warning_rgb = cv2.cvtColor(warning, cv2.COLOR_BGR2RGB)
    plt.imshow(warning_rgb)
    plt.axis('off')
    plt.show()
