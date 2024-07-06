from flask import Flask, request, jsonify, render_template
import pickle
import cv2
from skimage.feature import hog
import numpy as np
import base64

# Initialize Flask application
app = Flask(__name__)

# Load the trained Random Forest model using pickle
with open(r'random_forest_model_v1.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to extract HOG features from an image


def extract_hog_features(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
    return hog_features

# Endpoint to handle image upload and inference


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is actually an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Read the image file
            file_bytes = file.read()
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

            # Preprocess the image (resize, grayscale, HOG feature extraction)
            # Adjust resize dimensions as needed
            img_resized = cv2.resize(img, (128, 128))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            hog_features = extract_hog_features(img_gray)
            # Reshape features for prediction
            hog_features = np.array([hog_features])

            # Make prediction using the loaded model
            prediction = loaded_model.predict(hog_features)[0]

            # Prepare response
            # Convert prediction to string for JSON serialization
            response = {'prediction': str(prediction)}

            # Convert image to base64 string for HTML display
            img_str = base64.b64encode(file_bytes).decode('utf-8')

            return render_template('result.html', prediction=prediction, image=img_str)

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
