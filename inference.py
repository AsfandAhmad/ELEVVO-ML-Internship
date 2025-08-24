import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load model
model = tf.keras.models.load_model("outputs/checkpoints/best_model.keras", compile=False)

# Class names
class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3',
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40',
               '41', '42', '5', '6', '7', '8', '9']

# Get image path from command line
img_path = sys.argv[1]

# Load and preprocess image
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)
class_idx = np.argmax(pred)
confidence = float(np.max(pred))
predicted_class = class_names[class_idx]

print(f"Predicted class index: {class_idx}")
print(f"Predicted class name: {predicted_class}")
print(f"Confidence: {confidence:.2f}")

# Plot and save image with prediction
plt.imshow(image.load_img(img_path))
plt.axis("off")
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")

# Create outputs folder if not exists
os.makedirs("outputs/predictions", exist_ok=True)

# Save with timestamp so it doesn’t overwrite
out_path = f"outputs/predictions/pred_{predicted_class}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
plt.savefig(out_path)
print(f"Saved prediction image at: {out_path}")
