cluster.py---- It is used to load all images in the all folder, perform k-means clustering, K=8, and save the model to kmeans_model.joblib
kmeans_model.joblib---- Trained clustering model
prediction.py----It  loads the trained clustering model kmeans_model.joblib and predicts the images to be predicted in the test folder. If the category is determined to be '260 ° C', the fire_warning image will be displayed
all---- training set, image name contains temperature label
test---- Image to be predicted