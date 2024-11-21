import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# Directory for saving uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
MODEL_PATH = 'Boerhavia_Model_Final.keras'  # Adjust path if needed
model = load_model(MODEL_PATH)  # Use load_model from tensorflow.keras.models

# Define class labels
CLASS_LABELS = ['Boerhavia Diffusa', 'Boerhavia erecta']

# MySQL database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Replace with your MySQL username
    'password': '',  # Replace with your MySQL password
    'database': 'boerhavia_species'  # Replace with your database name
}

# Check if the file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(image_path, target_size=(180, 180)):
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Preprocess image for base64 encoded data
def preprocess_base64_image(image_data, target_size=(180, 180)):
    # Decode base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(image_data)).convert('RGB')  # Ensure image is in RGB format
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the main page with upload options
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/identify.html')
def identify():
    return render_template('identify.html')

@app.route('/species-detail.html')
def species_detail():
    return render_template('species-detail.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/scan.html')
def scan():
    return render_template('scan.html')

# Route for handling the image upload and displaying it on result page
@app.route('/result.html')
def result_page():
    filename = request.args.get('filename')
    predicted_species = request.args.get('species')
    file_path = url_for('static', filename=f'uploads/{filename}')
    return render_template('result.html', file_path=file_path, predicted_species=predicted_species)

# Route for handling the image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Ensure the 'static/uploads' directory exists
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create the directory if it doesn't exist

        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Preprocess the image and predict the species
        image = preprocess_image(file_path)
        predictions = model.predict(image)
        predicted_label = CLASS_LABELS[np.argmax(predictions)]

        # Redirect to results page with the uploaded image and prediction
        return redirect(url_for('result_page', filename=filename, species=predicted_label))

    return redirect(request.url)

# Route for handling the image scan (base64) and prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data sent in the request
    image_data = data['image']  # Extract the image data (base64)

    # Preprocess the image and predict the species
    image = preprocess_base64_image(image_data)
    predictions = model.predict(image)
    predicted_label = CLASS_LABELS[np.argmax(predictions)]  # Get the predicted class

    # Return the prediction as a JSON response
    return jsonify({'predicted_species': predicted_label})

# Route for saving Join Us form data into MySQL database
@app.route('/joinus', methods=['POST'])
def join_us():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')

    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = connection.cursor()
        query = "INSERT INTO join_us (name, email, phone) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, email, phone))
        connection.commit()
        print("Data saved successfully!")
    except Error as e:
        print(f"Database error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return redirect(url_for('index'))  # Ensure 'index' route exists



if __name__ == '__main__':
    app.run(debug=True)
