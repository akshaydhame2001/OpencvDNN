import cv2
import time
import numpy as np

# Load the COCO class names
with open('models/COCO.txt', 'r') as f:
    class_names = f.read().split('\n')

print(class_names)

# Get a different colors for each of the classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv2.dnn.readNet(model="")

