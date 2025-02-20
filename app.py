from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

# Flask uygulamasını başlat
app = Flask(__name__)

# Model yolu
model_path = 'Mobilenet_fine_last.h5'

# Gerekli özel katmanlar ve aktivasyon fonksiyonları
custom_objects = {'MobileNetV2': MobileNetV2, 'relu6': tf.nn.relu6}

# Modeli custom_objects ile yükleme
model = load_model(model_path, custom_objects=custom_objects)

# Sınıflar
classes = {
    0: ('nv', 'Nevus'),
    1: ('mel', 'Melanoma'),
    2: ('bkl', 'Seborrheic Keratosis'),
}

# Görüntüyü ön işleme fonksiyonu
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Image processing error: {e}")
        raise

# Ana sayfa rotası
@app.route('/')
def home():
    return "Flask API Çalışıyor!"

# Tahmin yapmak için API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = float(predictions[0][predicted_class[0]])

        predicted_class_name = classes[predicted_class[0]]

        return jsonify({
            'predicted_class': predicted_class_name[1],
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Flask uygulamasını başlat
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)