import os
import random
import pickle
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def display_images(directories, num_images=5):
    fig, axes = plt.subplots(len(directories), num_images,
                             figsize=(15, len(directories) * 2.5))
    fig.suptitle("Sample Images from Each Class", fontsize=16)

    for i, (label, directory) in enumerate(directories.items()):
        image_files = os.listdir(directory)
        random.shuffle(image_files)
        for j in range(num_images):
            image_path = os.path.join(directory, image_files[j])
            img = cv2.imread(image_path)
            axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{label} Image {j+1}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def extract_features(image):
    features = []

    # Low-level Vision features
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    features.extend(hist)

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray_image)
    hist_eq = hist_eq.flatten()
    features.extend(hist_eq)

    # Gray-scale transformation
    gray_flatten = gray_image.flatten()
    features.extend(gray_flatten)

    # Image Smoothing
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    smoothed_flatten = smoothed_image.flatten()
    features.extend(smoothed_flatten)

    # Mid-level Vision features
    # HOG features
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), visualize=False)
    features.extend(hog_features)

    # Edge Detection using Canny
    edges = cv2.Canny(gray_image, 100, 200)
    edges_flatten = edges.flatten()
    features.extend(edges_flatten)

    # SIFT features
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_image, None)
    if des is not None:
        des = des.flatten()
        features.extend(des)

    sift_feature_size = 128 * 30  # Adjust based on expected number of keypoints
    if len(features) > sift_feature_size:
        features = features[:sift_feature_size]
    else:
        features.extend([0] * (sift_feature_size - len(features)))

    return features


def load_and_extract_features(directories):
    X = []
    y = []
    for label, directory in directories.items():
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {filename}")
                continue

            img = cv2.imread(image_path)

            if img is None:
                print(f"Warning: Failed to load image {image_path}")
                continue

            img_resized = cv2.resize(img, (128, 128))
            features = extract_features(img_resized)
            X.append(features)
            y.append(label)
    return X, y


# Define the directories for each class
directories = {
    "Building": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Building",
    "Forest": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Forest",
    "Glacier": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Glacier",
    "Mountains": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Mountains",
    "Sea": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Sea",
    "Streets": "C:\\Users\\Muks\\Downloads\\dataset_1\\dataset_full\\Streets"
}

# Display sample images from each class
display_images(directories)

# Load and extract features from the dataset
X, y = load_and_extract_features(directories)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=1000, criterion='gini', max_depth=15, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Classification Accuracy:", accuracy)

# Evaluate model using various metrics
conf_matrix = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
class_report = classification_report(y_test, predictions)

# One-hot encode labels for ROC-AUC and log loss calculations
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_proba = rf_classifier.predict_proba(X_test)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovo')
log_loss_value = log_loss(y_test_bin, y_pred_proba)

# Print evaluation metrics
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)
print("Log Loss:", log_loss_value)
print("Classification Report:\n", class_report)

# Save the model to disk with pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)
print("Model saved successfully.")
