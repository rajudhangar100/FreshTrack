# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your pre-trained TensorFlow model
model = tf.keras.models.load_model("path_to_your_model.h5")

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0    # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(processed_image)
        result = "rotten" if prediction[0] > 0.5 else "fresh"

        freshnum = 0
        rottenum = 0
        for i in prediction:
            if ( i>0.5):
                rottenum+=1
            else:
                freshnum+=1

        freshindex = (freshnum/(freshnum+rottenum))
        
        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
