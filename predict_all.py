import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("outputs/checkpoints/best_model.keras", compile=False)

# Map class indices to class names
class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
               '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', 
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', 
               '41', '42', '5', '6', '7', '8', '9']

# Path to test dataset
test_root = "data/processed/test/"

# Counters for accuracy
total_images = 0
correct_predictions = 0

# Loop through each class folder
for class_folder in os.listdir(test_root):
    folder_path = os.path.join(test_root, class_folder)
    if not os.path.isdir(folder_path):
        continue
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        class_idx = np.argmax(pred)
        
        total_images += 1
        if class_names[class_idx] == class_folder:
            correct_predictions += 1
        
        print(f"Image: {img_path} -> Predicted: {class_names[class_idx]} (Actual: {class_folder})")

# Print overall accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"\nTotal images: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
