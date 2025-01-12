# HandControl_gesture_recognition


This project presents a hand gesture recognition system designed to control YouTube videos using machine learning. By using the Jester dataset and training a ResNet-101 model, the system can interpret ten distinct gestures in real-time through the webcam corresponding to various video control functionalities, such as play, pause, forward, rewind, and fullscreen. Depending on the hand gestures predicted, the corresponding keystrokes (keyboard shortcuts) will be sent to trigger actions on a computer. With a real-world accuracy of 75%, this project demonstrates the potential for touchless human-computer interaction (HCI) and sets the groundwork for further advancements.


# Features

Control YouTube videos using hand gestures.

Recognizes 10 gestures for actions such as play, pause, forward, rewind, and toggle fullscreen.

Implements ResNet-101 architecture for gesture recognition.

Focus on accessibility and hygienic interaction, especially relevant in healthcare and educational sectors.


# Model and Architecture

The model uses ResNet-101, a deep residual network for hierarchical feature extraction and overcoming vanishing gradients. Key configurations include:

Batch size: 32

Frames per sequence: 16

Dropout rate: 0.4

Learning rate: 0.05

Optimizer: Stochastic Gradient Descent (SGD)

Loss function: Categorical Cross Entropy

Epochs: 2 per segment

Metrics: Accuracy tracked during training, validation, and testing.


# Setup and Installation

Python 3.8+

TensorFlow

OpenCV

NumPy, Pandas, Matplotlib

# Installation

Clone the repository:

git clone https://github.com/hzahra55/HandControl_gesture_recognition.git

cd gesture_recognition

Install dependencies:

pip install -r requirements.txt

Download the Jester dataset and place it in the data folder.

# Challenges and Limitations

## Challenges

Initial convergence slowed by suboptimal hyperparameters.

Environmental noise (e.g., lighting variations) affected real-world performance.

## Limitations

Only 10 of 27 gesture classes were implemented.

Sensitive to lighting and background complexity.

Real-world accuracy limited to 60%.

# Future Work

Expand gesture classes to include all 27 from the Jester dataset.

Enhance real-world accuracy through improved preprocessing and data augmentation.

Incorporate temporal models (e.g., LSTMs) for sequential gesture recognition.

Explore applications in healthcare for touchless interaction in sterile environments.

# References

https://ieeexplore.ieee.org/abstract/document/9368658
https://github.com/MLphile/gesture_based_youtube_control
https://www.kaggle.com/datasets/toxicmender/20bn-jester?select=Train.csv
https://link.springer.com/article/10.1007/s42452-021-04897-7



