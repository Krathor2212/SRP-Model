import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io


app = Flask(__name__)

# Load the pre-trained TensorFlow model
model = load_model('nsfw_mobilenetv2.keras', compile=False)


# Define the label mapping
labels = {0: 'drawings', 1: 'hentai', 2: 'neutral', 3: 'porn', 4: 'sexy'}

# Define a route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    try:
        # Convert the uploaded file to a file-like object
        image = load_img(io.BytesIO(file.read()), target_size=(224, 224))  # Adjust target_size as per your model
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize if required by the model

        # Perform prediction
        predictions = model.predict(image)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        label = labels.get(class_index, "Unknown")

        return jsonify({
            'class_index': int(class_index),
            'label': label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's port, default to 5000 locally
    app.run(debug=True, host='0.0.0.0', port=port)
