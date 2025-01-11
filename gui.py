# import os
# import numpy as np
# import cv2
# import time  # For adding a delay
# from pynput.keyboard import Key, Controller
# from keras.models import load_model
# from lib.data_loader import DataLoader  

# # Define constants
# ROOT = r'D:/python/gesture_control/dataset'  
# MODEL_PATH = 'D:/python/gesture_control/model/resnet_101_final.keras'  # Path to trained model
# WIDTH = 96
# HEIGHT = 64
# N_FRAMES = 16

# # Define gesture-to-action mapping
# HAND_GESTURE_ACTION_MAPPING = {
#     'Swiping Left': 'fast forward 10 seconds',
#     'Swiping Right': 'rewind 10 seconds',
#     'Swiping Down': 'previous video',
#     'Swiping Up': 'next video',
#     'Sliding Two Fingers Down': 'decrease volume',
#     'Sliding Two Fingers Up': 'increase volume',
#     'Thumb Down': 'mute / unmute',
#     'Thumb Up': 'enter / exit full screen',
#     'Stop Sign': 'play / pause',
#     'No gesture': 'no action'
# }

# # Load dataset mappings
# def load_dataset_mappings():
#     labels_csv_path = os.path.join(ROOT, 'labels_extracted.csv')
#     train_csv_path = os.path.join(ROOT, 'train_extracted.csv')
#     val_csv_path = os.path.join(ROOT, 'validation_extracted.csv')
#     data = DataLoader(labels_csv_path, train_csv_path, val_csv_path)
#     return data.int_to_label

# # Initialize the camera and model
# def initialize_camera_and_model():
#     print('Loading model...')
#     model = load_model(MODEL_PATH, compile=False)
#     print('Model loaded successfully')
#     cam = cv2.VideoCapture(0)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
#     return model, cam

# # Gesture detection and action
# def detect_gesture_and_take_action(model, cam, int_to_label):
#     buffer = []
#     keyboard = Controller()

#     # Initialize default gesture and action
#     gesture = "No gesture"
#     action = "no action"

#     while cam.isOpened():
#         ret, frame = cam.read()
#         if not ret:
#             print("Failed to capture frame. Exiting.")
#             break

#         # Preprocess frame
#         image = cv2.resize(frame, (WIDTH, HEIGHT))
#         image = image / 255.0
#         buffer.append(image)

#         # Predict gesture when buffer is full
#         if len(buffer) == N_FRAMES:
#             input_buffer = np.expand_dims(buffer, 0)
#             predicted_value = np.argmax(model.predict(input_buffer, verbose=0))
#             buffer = []

#             # Get the gesture and perform action
#             gesture = int_to_label.get(predicted_value, "No gesture")
#             action = HAND_GESTURE_ACTION_MAPPING.get(gesture, "no action")
#             print(f"Gesture: {gesture}, Action: {action}")
#             if gesture != "No gesture":  # Skip wait if it's "No gesture"
#                 perform_action(predicted_value, keyboard)
#                 time.sleep(1)  # Add a 1-second delay for other gestures

#         # Display the frame with overlay text
#         cv2.putText(frame, f"{gesture} -> {action}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
#         cv2.imshow('Gesture Control', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cam.release()
#     cv2.destroyAllWindows()

# # Perform action based on gesture
# def perform_action(predicted_value, keyboard):
#     actions = {
#         0: lambda: keyboard.tap('l'),  # Swiping Left
#         1: lambda: keyboard.tap('j'),  # Swiping Right
#         2: lambda: keyboard.tap('p'),  # Swiping Down
#         3: lambda: keyboard.tap('n'),  # Swiping Up
#         4: lambda: keyboard.tap(Key.down),  # Sliding Two Fingers Down
#         5: lambda: keyboard.tap(Key.up),  # Sliding Two Fingers Up
#         6: lambda: keyboard.tap('m'),  # Thumb Down
#         7: lambda: keyboard.tap('f'),  # Thumb Up
#         8: lambda: keyboard.tap('k'),  # Stop Sign
#         9: lambda: None  # No gesture
#     }
#     if predicted_value in actions:
#         actions[predicted_value]()

# # Main function
# def main():
#     int_to_label = load_dataset_mappings()
#     model, cam = initialize_camera_and_model()
#     detect_gesture_and_take_action(model, cam, int_to_label)

# if __name__ == "__main__":
#     main()



import os
import numpy as np
import cv2
import time  # For adding a delay
from pynput.keyboard import Key, Controller
from keras.models import load_model
from lib.data_loader import DataLoader  

# Define constants
ROOT = r'D:/python/gesture_control/dataset'  
MODEL_PATH = 'D:/python/gesture_control/model/resnet_101_final.keras'  # Path to trained model
WIDTH = 96
HEIGHT = 64
N_FRAMES = 16

# Define gesture-to-action mapping
HAND_GESTURE_ACTION_MAPPING = {
    'Swiping Left': 'fast forward 10 seconds',
    'Swiping Right': 'rewind 10 seconds',
    'Swiping Down': 'previous video',
    'Swiping Up': 'next video',
    'Sliding Two Fingers Down': 'decrease volume',
    'Sliding Two Fingers Up': 'increase volume',
    'Thumb Down': 'mute / unmute',
    'Thumb Up': 'enter / exit full screen',
    'Stop Sign': 'play / pause',
    'No gesture': 'no action'
}

# Load dataset mappings
def load_dataset_mappings():
    labels_csv_path = os.path.join(ROOT, 'labels_extracted.csv')
    train_csv_path = os.path.join(ROOT, 'train_extracted.csv')
    val_csv_path = os.path.join(ROOT, 'validation_extracted.csv')
    data = DataLoader(labels_csv_path, train_csv_path, val_csv_path)
    return data.int_to_label

# Initialize the camera and model
def initialize_camera_and_model():
    print('Loading model...')
    model = load_model(MODEL_PATH, compile=False)
    print('Model loaded successfully')
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    return model, cam

# Gesture detection and action
def detect_gesture_and_take_action(model, cam, int_to_label):
    buffer = []
    keyboard = Controller()

    # Initialize default gesture and action
    gesture = "No gesture"
    action = "no action"

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Preprocess frame
        image = cv2.resize(frame, (WIDTH, HEIGHT))
        image = image / 255.0
        buffer.append(image)

        # Predict gesture when buffer is full
        if len(buffer) == N_FRAMES:
            input_buffer = np.expand_dims(buffer, 0)
            predicted_value = np.argmax(model.predict(input_buffer, verbose=0))
            buffer = []

            # Get the gesture and perform action
            gesture = int_to_label.get(predicted_value, "No gesture")
            action = HAND_GESTURE_ACTION_MAPPING.get(gesture, "no action")
            print(f"Gesture: {gesture}, Action: {action}")
            if gesture != "No gesture":
                perform_action(predicted_value, keyboard)
                time.sleep(1)  # Add a 1-second delay for other gestures

        # Display the frame with overlay text
        cv2.putText(frame, f"{gesture} -> {action}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
        cv2.imshow('Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Perform action based on gesture
def perform_action(predicted_value, keyboard):
    actions = {
        0: lambda: keyboard.tap('l'),  # Swiping Left
        1: lambda: keyboard.tap('j'),  # Swiping Right
        6: lambda: keyboard.tap('m'),  # Thumb Down
        8: lambda: keyboard.tap('k'),  # Stop Sign (play/pause)
    }
    if predicted_value in actions:
        actions[predicted_value]()

# Main function
def main():
    int_to_label = load_dataset_mappings()
    model, cam = initialize_camera_and_model()
    detect_gesture_and_take_action(model, cam, int_to_label)

if __name__ == "__main__":
    main()
