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

```
![Screenshot 2025-01-07 163829](https://github.com/user-attachments/assets/9e9230b4-9a21-4072-96c9-049c247a293a)

![IMG-20250107-WA0008](https://github.com/user-attachments/assets/a371f08c-8874-4c42-ab43-0aa1486cfd78)

- Red - alarm toggle, toggle object detection and alarm on/off
- Blue - date / time
- Purple - motion detected alert 
- Green - type of object detected
- Yellow - camera view
