# DIP-Project


# Age and Gender Prediction Using OpenCV and Deep Learning

This project utilizes OpenCV and deep learning to predict the age and gender of individuals from images or video streams captured by a camera.

## Overview

The script uses a pre-trained deep learning model for face detection and two separate pre-trained models for gender and age prediction. It detects faces in real-time through the webcam feed or provided images, then predicts the gender and age of the detected faces.

## Prerequisites

- Python 3
- OpenCV (cv2)
- Numpy

## Installation

1. Clone the repository:

git clone https://github.com/Gautham-Nadimpalli/DIP-Project.git
cd age-gender-prediction


2. Install the required dependencies:

pip install opencv-python numpy


## Usage

1. Run the script:

python age_gender_prediction_video_model1.py

2. The webcam feed will open, and faces will be detected and labeled with predicted gender and age.

3. Press the `Esc` key to exit the program.

## Configuration

- Ensure the pre-trained model files (`age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`) are present in the same directory as the script.

## Notes
- The face detection is based on the Haar Cascade classifier (`haarcascade_frontalface_default.xml`).Please ensure it is present in the project directory or provide the correct path in the code.
































