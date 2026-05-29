# Facial Recognition Starter Small Project

<img width="1760" height="1045" alt="image" src="https://github.com/user-attachments/assets/a6979900-715c-408c-a0c4-fdfc0db135de" />
<img width="1765" height="1048" alt="image" src="https://github.com/user-attachments/assets/451332e2-36b8-44db-802c-d1437d773163" />
<img width="795" height="496" alt="image" src="https://github.com/user-attachments/assets/274809e9-540b-4abf-9d0d-d1b3e3ef93bc" />



Interactive computer vision project using Python, OpenCV, and MediaPipe to apply real time visual effects inside a hand gesture controlled region on a webcam.

The application detects hands and finger positions using MediaPipe, allowing the user to dynamically create and resize an effect area using both hands. Effects can be cycled in real time using pinch gestures.
Features

- Real time webcam processing
- Hand and finger tracking using MediaPipe
- Pinch controlled effect switching
- Dynamic bounding box resizing using hand positions
- Multiple custom OpenCV visual effects
- Fullscreen interactive display


Current Effects

- None
- Glitch
- Chromatic Aberration
- Sketch
- Thermal Vision
- Cartoon
- Invert


How It Works

- Two wrists are used as opposite corners of a rectangle.
- The rectangle becomes the effect region.
- Any selected visual filter is applied only inside this region.
- Pinching the thumb and index finger together cycles to the next effect.
- The rectangle updates in real time as hands move.

Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

 Installation

1. Clone the repository

```bash
git clone https://github.com/connor56576/facial-recognition-starter-small-project.git
cd facial-recognition-starter-small-project
```

2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

Windows (Git Bash)

```bash
source venv/Scripts/activate
```
Windows (CMD)

```bash
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

Running the Project

```bash
python main.py
```

## Future Improvements

- FPS counter (Done)
- Interface
- Additional visual effects (Doing)
- Modular effect system (Done)
- Face recognition integration
- GPU acceleration
- Custom gesture mapping
- Save or load presets
- YOLO 

 Notes

This project was built primarily as an starter computer vision project using OpenCV and MediaPipe.

Many of the effects were created experimentally to explore different image manipulation techniques and performance tradeoffs in live video processing.
