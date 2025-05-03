import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model("vegetable.h5")

# Load precautions Excel file
df_precautions = pd.read_excel('precautions veg.xlsx')

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'image' not in request.files:
            return "No file part in the request"

        file = request.files['image']

        # If the user does not select a file
        if file.filename == '':
            return "No file selected"

        # If a valid file is selected, process it
        if file and allowed_file(file.filename):
            # Save the file securely
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Load and preprocess the image
            img = load_img(file_path, target_size=(128, 128))  # Resize to 128x128
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)  # Add batch dimension

            # Make prediction
            try:
                prediction = model.predict(x)
                pred = np.argmax(prediction, axis=-1)[0]  # Get predicted class index
                precaution = df_precautions.iloc[pred]['Precautions']
                return render_template('result.html', precaution=precaution)
            except Exception as e:
                return f"Error in prediction: {str(e)}"
        else:
            return "File type not allowed"
    return render_template('predict.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
