# Railway Track Surveillance

The aim is to monitor and ensure the safety and maintenance of railway tracks, built with Python, Flask, and OpenCV. It detects motion using a webcam, highlights detected objects using YOLOv3, and provides a live video feed on a web interface.

## Features

- Real-time monitoring of railway tracks.
- Detection of anomalies and potential issues.
- Object detection with YOLOv3 (choose any YOLO model, Preferred V3 for weak system)
- Web interface for live video feed and status display
- Adjustable alarm mode to enable or disable motion detection
- User-friendly interface for displaying surveillance data.

---

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/abhishek0450/Railway-Track-Surveillance.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Railway-Track-Surveillance
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the project, execute the following command:
```sh
python app.py
