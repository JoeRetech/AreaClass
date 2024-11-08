from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploader/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SIZE = 24

# Load your pre-trained model
model = keras.models.load_model(r'model/model.h5')
categories = ['Clean', 'Dirty']

def send_email_with_image(prediction, location, image_path):
    sender = "amlananshu6a@gmail.com"
    receiver = "srinithivminiproject@gmail.com"
    password =  "qizzyggtirnxnmmn"

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = f'Prediction: {prediction}'

    # Email body
    body = f'Prediction: {prediction}\nLocation: {location}'
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename='skin_image.png')
        msg.attach(img)

    # Send email via SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    location = request.form.get('location', 'Unknown')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
    file.save(file_path)

    try:
        # Process the image for prediction
        nimage = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if nimage is None:
            return jsonify({'error': 'Image could not be read.'})

        image = cv2.resize(nimage, (SIZE, SIZE)) / 255.0
        prediction = model.predict(np.array(image).reshape(-1, SIZE, SIZE, 1))
        pclass = np.argmax(prediction)
        predicted_label = categories[int(pclass)]

        # Send email if prediction is 'Dirty'
        if predicted_label == 'Dirty':
            send_email_with_image(predicted_label, location, file_path)

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})


