import base64
import io
import numpy as np
import math

import tensorflow as tf
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['API_KEY'] = 'oEfC3PglPKoCg1jDa833awsnTLoCxWSjbumTypmSEbNgWAincCp00DkcFEw45JznC6Cou73GrU07VieU01ktlsckPyqlWoSU75Bf'

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

#Load Model
detection_model_path_1 = 'model/ImageDetection1.pt'
detection_model_path_2 = 'model/ImageDetection2.pt'
model_path = 'model/optiguard_model_v1.h5'
CNN = tf.keras.models.load_model(model_path)


#OBJECT DETECTION
def objectDetection(image, model_path):
    detector = ObjectDetection(model_path)
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

    return cropped_images


class ObjectDetection:
    def __init__(self, model_path):
        self.capture = None
        self.model = None
        self.CLASS_NAMES_DICT = None
        self.model_path = model_path

    def set_capture(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names

    def load_model(self):
        model = YOLO(self.model_path)
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
  
  

# ROUTES
@app.route('/')
def index():
    return jsonify({'message': 'Hello World!'})

  
@app.route('/predict', methods=['POST'])
def predict():
    # Check for api key
    api_key = request.headers.get('x-api-key')
    if api_key != app.config['API_KEY']:
        return jsonify({'success': False, 'error': 'API Key tidak valid'}), 401
    
    # Parse JSON request body
    data = request.get_json()
    if 'fundus_image' not in data:
        return jsonify({'success': False, 'error': 'Fundus image tidak ditemukan'}), 400

    fundus_image = data['fundus_image']
    
    try:
        # Decode base64 image blob
        image_bytes = base64.b64decode(fundus_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        height, width, _ = image_np.shape
        print(f"Width: {width}, Height: {height}")
        
        if (height/width) > 1.5 or (width/height) > 1.5:    
            detected_image_1 = objectDetection(image_np, detection_model_path_1)
            if len(detected_image_1) == 0:
                print('Object detection 1 failed')
                return jsonify({'success': True, 'message': 'Tidak ada fundus yang terdeteksi'}), 200
            
            image_np = detected_image_1[0]
        
        detected_image_2 = objectDetection(image_np, detection_model_path_2)
        if len(detected_image_2) == 0:
            print('Object detection 2 failed')
            return jsonify({'success': True, 'message': 'Tidak ada fundus yang terdeteksi'}), 200
        
        input_image = detected_image_2[0]
        input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        output_image = mask_ellipse(input_image)
        output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

        predicted_class = ConvolutionalNN(output_image)
        print(predicted_class)
        
        # Convert output_image to base64
        _, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'data': {
                'predicted_class': predicted_class, 
                "cropped_image": output_image_base64
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)