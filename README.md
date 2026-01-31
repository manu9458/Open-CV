# Industrial Grade AI  Monitoring System

This project is a modular, high-performance motion detection system using OpenCV.

## Modules

- **main.py**: The entry point of the application.
- **camera.py**: Handles threaded camera capture for better performance.
- **surveillance.py**: Core logic for motion detection, background subtraction, and trajectory tracking.
- **logger.py**: Handles logging of motion events to CSV.
- **requirements.txt**: List of dependencies.

## Installation & Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate Environment**:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python main.py
```

## Features

- **Object Detection (YOLOv8)**: Uses deep learning to identify 80+ types of objects (People, Cars, Phones, etc.).
- **Real-Time Classification**: Displays class names and confidence scores on screen.
- **Activity Logging**: Logs detected object types to CSV.
- **Threaded Performance**: Optimized for higher FPS.

