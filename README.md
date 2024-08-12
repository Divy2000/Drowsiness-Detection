# Student Drowsiness Detection System

This project aims to detect student drowsiness using real-time camera feeds or saved videos. It employs a custom CNN model and efficient data handling techniques to achieve high accuracy in drowsiness detection.

## Project Overview

- **Image Extraction**: Extracted 57,488 images from ultraLDD and Google Drive datasets.
- **Preprocessing**: Applied image masking to focus on eyes and mouth using OpenCV, dlib, and faceutils.
- **Model**: Custom CNN model inspired by VGG network architecture.
- **Training**: Utilized TensorFlow's `ImageDataGenerator` and a custom data generator for efficient training.
- **Prediction**: Real-time drowsiness detection with overlay on live camera feed or saved videos.

## Dataset

1. **Image Extraction**:
   - Extract images from video files using OpenCV.
   - Map images with their respective labels (alert, low, vigilant, drowsy).

2. **Data Preparation**:
   - Combine images from the ultraLDD dataset and Google Drive dataset (total: 57,488 images).

## Model Architecture

- **CNN Structure**:
  - 3 Convolutional Layers with Max Pooling
  - 2 Dense Layers

## Training Pipeline

1. **Data Loading**:
   - Use TensorFlow's `ImageDataGenerator` for loading images on-the-fly.
   - Implement a custom data generator using TensorFlow’s `tf.data` API for improved performance.

2. **Training**:
   - Split data into training and validation datasets (80:20 ratio).
   - Achieve acceptable accuracy within 5 epochs.

3. **Results**:
   - Training Loss: 0.2562
   - Training Accuracy: 92.59%
   - Validation Loss: 0.2447
   - Validation Accuracy: 93.01%

## Prediction Pipeline

1. **Real-Time Prediction**:
   - Predict from live camera feeds or saved videos.
   - Display drowsiness detection results with an overlay on the screen.

2. **Usage**:
   - For live feed predictions, ensure your camera is connected and configured.
   - For video file predictions, specify the video file path in the script.
