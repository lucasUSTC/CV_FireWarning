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

# repalce your file path
folder_path = '/content/sample_data/all'

images = load_images_from_folder(folder_path)

hsv_features = extract_hsv_features(images)

num_clusters = 8  # 8
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(hsv_features)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Save the trained KMeans model
model_path = 'kmeans_model.joblib'
joblib.dump(kmeans, model_path)


plt.figure(figsize=(10, 10))
for i in range(len(hsv_features)):
    plt.scatter(hsv_features[i, 0], hsv_features[i, 1], s=20, color=plt.cm.jet(labels[i] / num_clusters),alpha=0.4, linewidths=0.5 )# marker='*'

plt.title('Clustering of Images based on H-S Features')
plt.xlabel('Hue')
plt.ylabel('Saturation')

# Adjusted class labels and colors order
class_labels = ['25℃', '50℃', '100℃', '150℃', '200℃', '240℃', '250℃', '260℃']
original_order = ['260℃', '50℃', '150℃', '240℃', '100℃', '200℃', '25℃', '250℃']
color_order = [plt.cm.jet(i / num_clusters) for i in range(num_clusters)]
order_mapping = {label: color for label, color in zip(original_order, color_order)}
adjusted_colors = [order_mapping[label] for label in class_labels]

for color, label in zip(adjusted_colors, class_labels):
    plt.scatter([], [], color=color, label=label)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

plt.show()