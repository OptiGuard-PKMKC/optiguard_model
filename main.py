import base64
import os
import numpy as np
import math

import tensorflow as tf
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['API_KEY'] = 'oEfC3PglPKoCg1jDa833awsnTLoCxWSjbumTypmSEbNgWAincCp00DkcFEw45JznC6Cou73GrU07VieU01ktlsckPyqlWoSU75Bf'

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

#Load Model
object_detection = YOLO('model/ImageDetection2.pt')
model_path = 'model/optiguard_model_v1.h5'
CNN = tf.keras.models.load_model(model_path)


@app.route('/')
def index():
    return jsonify({'message': 'Hello World!'})

#OBJECT DETECTION
def objectDetection(image):
    detector = ObjectDetection()
    detector.set_capture(image)

    cropped_images = []

    results = detector.predict(image)

    if len(results) > 0:
        r = results[0]
        if len(r.boxes) > 0:
            box = r.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = detector.crop_image(image, x1, y1, x2, y2)
            cropped_images.append(cropped_img)

            # cv2_imshow(cropped_img)

        # img_with_boxes = detector.plot_boxes(results, img)
        # cv2_imshow(img_with_boxes)

    # cv2.destroyAllWindows()
    return cropped_images

class ObjectDetection:
    def __init__(self):
        self.capture = None
        self.model = None
        self.CLASS_NAMES_DICT = None

    def set_capture(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names

    def load_model(self):
        model = object_detection
        return model

    def predict(self, img):
        results = self.model(img)

        return results

    def crop_image(self, img, x1, y1, x2, y2):
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img

    def plot_boxes(self, results, img):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5:
                    cv2.putText(img, f'{currentClass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return img
    

#MASKING IMAGE
def mask_ellipse(image):
    image = image.convert("RGBA")

    width, height = image.size
    
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    
    result = Image.new("RGBA", (width, height))
    result.paste(image, (0, 0), mask=mask)
    
    max_size = max(width, height)
    
    background = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 255))
    
    offset_x = (max_size - width) // 2
    offset_y = (max_size - height) // 2
    background.paste(result, (offset_x, offset_y), mask=mask)
    
    final_image = Image.alpha_composite(background, background)
    
    return final_image.convert("RGB")

#KLASIFIKASI CNN
import cv2
import numpy as np

def ConvolutionalNN(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (324, 324))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    model = CNN

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]

    # plt.figure(figsize=(6, 6))
    # plt.imshow(img)
    # plt.title(f"Predicted: {class_names[predicted_class_index]}", fontsize=12)
    # plt.axis("off")
    # plt.show()

    # return class_names[predicted_class_index]

    return predicted_class
  
@app.route('/predict', methods=['POST'])
def predict():
    # Check for api key
    api_key = request.headers.get('x-api-key')
    if api_key != app.config['API_KEY']:
        return jsonify({'error': 'API Key tidak valid'}), 401
    
    # Parse JSON request body
    data = request.get_json()
    if 'fundus_image' not in data:
        return jsonify({'error': 'Fundus image tidak ditemukan'}), 400

    fundus_image = data['fundus_image']
    
    try:
        # Decode base64 image blob
        image_bytes = base64.b64decode(fundus_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
        detected_images = objectDetection(image_np)
        input_image = detected_images[0]
        input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        output_image = mask_ellipse(input_image)
        output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

        predicted_class = ConvolutionalNN(output_image)
        print(predicted_class)

        return jsonify({'predicted_class': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)