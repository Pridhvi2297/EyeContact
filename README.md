# EyeContact

A computer vision application that detects and analyzes eye contact in images or video streams.

## Overview

EyeContact is a tool that uses computer vision and machine learning techniques to detect faces, identify eye regions, and determine if a person is making eye contact with the camera. This can be useful for applications in human-computer interaction, attention analysis, and social interaction studies.

## Features

- Real-time face detection
- Eye region identification
- Eye contact/gaze direction analysis
- Support for both image and video processing
- Visual feedback on detection results

## Requirements

### Hardware
- Camera (webcam for real-time applications)
- Computer with sufficient processing power for real-time video analysis

### Software Dependencies
- Python 3.7+
- OpenCV (cv2)
- NumPy
- dlib
- face_recognition
- imutils
- matplotlib (for visualization)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pridhvi2297/EyeContact.git
   cd EyeContact
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt file, install the dependencies manually:
   ```bash
   pip install opencv-python numpy dlib face_recognition imutils matplotlib
   ```

   Note: Installing dlib might require additional system dependencies. On Ubuntu/Debian:
   ```bash
   sudo apt-get install cmake build-essential
   ```

## Usage

### Running the application

1. For real-time webcam analysis:
   ```bash
   python eye_contact_detector.py --source webcam
   ```

2. For processing a video file:
   ```bash
   python eye_contact_detector.py --source video --path path/to/video.mp4
   ```

3. For processing an image:
   ```bash
   python eye_contact_detector.py --source image --path path/to/image.jpg
   ```

### Controls

- Press 'q' to quit the application
- Press 's' to save the current frame (when processing video)

## How It Works

1. Face detection using dlib's HOG-based face detector or a deep learning-based detector
2. Facial landmark detection to identify eye regions
3. Eye gaze direction estimation using geometric features or a trained model
4. Classification of whether the person is making eye contact with the camera

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dlib library for face detection and facial landmark prediction
- OpenCV community for computer vision tools
- Contributors and maintainers of the face_recognition library
