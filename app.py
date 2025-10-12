# app.py
from flask import Flask, render_template, request,jsonify
import os
import torch
from src.deforestation.model import load_deforestation_model
from src.deforestation.predict import predict_deforestation
from src.wildfire.predict import predict_wildfire

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "models/deforestation_model.pth"
model = load_deforestation_model(MODEL_PATH, device)


# Routes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/deforestation')
def deforestation_page():
    return render_template('deforestation.html')

@app.route('/predict_deforestation', methods=['POST'])
def predict_deforestation_route():
    if 'image' not in request.files:
        return render_template('error.html', message="No image uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', message="No file selected")

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict
    label, confidence = predict_deforestation(model, filepath, device)
    confidence = round(confidence * 100, 2)
    filename = file.filename  # ✅ just the file name
    return render_template('results.html',
                           image_filename=filename,
                           label=label,
                           confidence=confidence)

@app.route('/wildfire')
def wildfire_page():
    return render_template('wildfire.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input
        required_fields = ['latitude','longitude','avgtemp_c','total_precip_mm','avg_humidity','pressure_in','wind_kph']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        result = predict_wildfire(data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
