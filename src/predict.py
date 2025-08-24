import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("outputs/checkpoints/best_model.keras", compile=False)

# Map class indices to class names (optional)
class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
               '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', 
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', 
               '41', '42', '5', '6', '7', '8', '9']

# Predict a single image
img_path = "data/processed/test/0/img001.png"  # replace with your image
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
class_idx = np.argmax(pred)
print("Predicted class index:", class_idx)
print("Predicted class name:", class_names[class_idx])
