import argparse

import cv2.dnn
import numpy as np
import time

with open("models/coco80.txt", "r") as f:
    class_list = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(class_list), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draw bounding boxes on the input image based on the provided arguments.

    Args:
        img (np.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{class_list[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX("models/yolov8s.onnx")

    # Set backend and target (optional, but good practice)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    prev_time = time.time()

    while True:
        ret, original_image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            min_score, max_score, min_class_loc, (x, max_class_index) = cv2.minMaxLoc(classes_scores)
            if max_score >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                    outputs[0][i][2],  # width
                    outputs[0][i][3],  # height
                ]
                boxes.append(box)
                scores.append(max_score)
                class_ids.append(max_class_index)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        if result_boxes is not None:
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": class_list[class_ids[index]],
                    "confidence": scores[index],
                    "box": box,
                    "scale": scale,
                }
                detections.append(detection)
                draw_bounding_box(
                    original_image,
                    class_ids[index],
                    scores[index],
                    round(box[0] * scale),
                    round(box[1] * scale),
                    round((box[0] + box[2]) * scale),
                    round((box[1] + box[3]) * scale),
                )

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        cv2.putText(original_image, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the image with bounding boxes
        cv2.imshow("Webcam Feed", original_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()