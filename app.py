# app.py

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

# Load the pre-trained model
model = load_model('covid.h5')

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = (model.predict(x) > 0.5).astype(int)
    if(preds==0):
      return "Covid"
    else:
     return "Normal"

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        file_path = f"{file.filename}"
        file.save(file_path)
        prediction = predict_image(file_path)
        return f"The uploaded image is predicted as: {prediction}"
    else:
        return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)



