from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Izinkan CORS for mobile app communication

# Mengatur path untuk model
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, 'student_stresslevel.pkl')

# Memuat model hanya sekali saat aplikasi berjalan
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Test apakah model memiliki metode predict
print(type(model))  # Harus menunjukkan model scikit-learn

@app.route('/')
def home():
    return "Flask API is running"

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah data ada dalam format JSON
    try:
        data = request.get_json()

        # Pastikan semua input ada dalam data yang diterima
        required_features = [
            'anxiety_level', 'self_esteem', 'mental_health_history',
            'depression', 'headache', 'sleep_quality',
            'noise_level', 'living_conditions', 'basic_needs',
            'academic_performance', 'study_load', 
            'teacher_student_relationship', 'social_support'
        ]
        
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Ambil semua inputan dari data
        features = [
            data['anxiety_level'], data['self_esteem'], data['mental_health_history'],
            data['depression'], data['headache'], data['sleep_quality'],
            data['noise_level'], data['living_conditions'], data['basic_needs'],
            data['academic_performance'], data['study_load'], 
            data['teacher_student_relationship'], data['social_support']
        ]
        
        # Prediksi dengan model
        prediction = model.predict([features])

        # Mengembalikan hasil prediksi
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)