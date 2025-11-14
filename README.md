# AI Virtual Painter 🎨

This is an application that enables one to vitually paint in the air using their fingers. It is developed in python on openCV and Mediapipe. So go ahead and recreate your imaginations in the air !

![Alt text](paint.gif)

## ✨ Features

  * **Virtual Brush:** Draw on the canvas using the tip of your right index finger.
  * **Gesture-Based Mode Control:** Instantly switch between `Draw` and `Standby` modes using your left hand. A **closed fist** activates drawing, while an **open hand** allows for selection and navigation.
  * **AI Shape Correction:** Automatically detect and perfect your hand-drawn strokes into clean geometric shapes like circles, squares, triangles, hearts, pentagons, and more. This can be toggled on/off with the 'A' key.
  * **AI Gesture-Forced Shapes:** Use your **left hand** to make specific gestures (e.g., an "OK" sign for a circle) to pre-select a shape *before* you draw it, ensuring a perfect result.
  * **AI Letter Recognition:** Toggle a special "Letter Mode" (`L` key) to recognize your hand-drawn letters (A-Z) and type them cleanly onto the canvas.
  * **Intuitive Toolbar:** Select colors and brush sizes by hovering your right index finger over the UI elements while in `Standby` mode (open left hand).
  * **Custom Gesture Training:** Includes a `train.py` script to train, save, and load your own custom hand gestures from scratch.
  * **Canvas Controls:** Use keyboard shortcuts to **Clear** (`c`), **Save** (`s`), and **Quit** (`q`).
  * **Dynamic UI:** Features a toggleable color palette, FPS counter, and a help guide for shape-forcing gestures (`G` key).

## 💻 Tech Stack

  * **Python**
  * **OpenCV:** For camera feed, UI windows, and image processing.
  * **MediaPipe:** For real-time hand tracking and landmark detection.
  * **NumPy:** For numerical operations on stroke data and landmark coordinates.
  * **pickle:** For serializing and saving trained gesture data.

## 🚀 Getting Started

1.  Clone the repository:
    ```bash
    git clone https://github.com/darthdaenerys/Virtual-Painter.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Virtual-Painter
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the main application:
    ```bash
    python paint.py
    ```
5.  (Optional) To train your own gestures:
    ```bash
    python train.py
    ```
