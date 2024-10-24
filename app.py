from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile
import os

app = Flask(__name__)

# Load your pre-trained TensorFlow model
model = tf.keras.models.load_model("fruit_classifier.h5")

# Preprocess the image
def prepare_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Load image and resize to 224x224
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert the image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    img_array = tf.keras.apllications.vgg16.preprocess_input(img_array)  # Preprocess the image for VGG16
    return img_array

# API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    try:
        # Create a temporary directory to store extracted images
        os.makedirs("temp", exist_ok=True)

        # Extract ZIP file
        zip_file_path = os.path.join("temp", file.filename)
        file.save(zip_file_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("temp")

        freshnum = 0
        rottenum = 0

        # Process each image in the extracted folder
        for image_file in os.listdir("temp"):
            image_path = os.path.join("temp", image_file)
            if os.path.isfile(image_path):
                processed_image = prepare_image(image_path)

                # Make predictions
                prediction = model.predict(processed_image)
                print(prediction)

                class_names=['Rotten','Fresh']
                predicted_class = class_names[np.argmax(prediction)]

                if predicted_class== "Rotten":
                    rottenum += 1
                else:
                    freshnum += 1

        # Calculate freshindex
        total_images = freshnum + rottenum
        freshindex = freshnum / total_images if total_images > 0 else 0

        # Clean up temporary files
        for image_file in os.listdir("temp"):
            os.remove(os.path.join("temp", image_file))
        os.rmdir("temp")

        return jsonify({
            "fresh_count": freshnum,
            "rotten_count": rottenum,
            "freshindex": freshindex
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
