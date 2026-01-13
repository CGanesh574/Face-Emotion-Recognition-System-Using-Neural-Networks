
# 🎭 Face & Emotion Recognition System

A **real-time face recognition and emotion detection system** built using **Python, OpenCV, and PyTorch**.
The system identifies individuals and classifies facial emotions simultaneously from a live webcam feed.

---

## 🌟 Key Features

### 🔍 Face Recognition

* Real-time multi-face detection
* Face recognition
* SVM classifier for improved accuracy
* Automatic face sample collection (50 images per person)
* Confidence-based recognition
* Model save & load support

### 😊 Emotion Recognition

* 7 emotion classes:

  * Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
* Custom CNN built with PyTorch
* Real-time emotion prediction
* Confidence thresholding for reliable output

### 🎯 Advanced Detection

* Face direction detection (Left / Right / Forward)
* Eye tracking-based gaze estimation
* Glasses detection
* Multi-face processing in a single frame

### 💻 User Interface

* Tkinter-based GUI
* Live camera feed (640×480)
* Add new faces dynamically
* Start/Stop recognition controls
* Real-time results display

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.7+
* **Computer Vision:** OpenCV
* **Deep Learning:** PyTorch
* **Machine Learning:** scikit-learn
* **GUI:** Tkinter
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## 🧠 System Workflow

### Face Recognition Pipeline

1. Webcam input
2. Haar Cascade face detection
3. Grayscale conversion & histogram equalization
4. Face resizing (100×100)
5. Feature extraction
6. SVM classification with confidence score

### Emotion Recognition Pipeline

1. Face crop
2. Resize to 48×48 grayscale
3. Normalization
4. CNN-based emotion classification
5. Softmax confidence output

---

## 📁 Project Structure

```
ml_project/
│
├── app.py                     # GUI application
├── emotion_recognition.py     # Emotion detection module
├── face_recognition.py        # Face recognition module
├── train.py                   # Model training scripts
├── requirements.txt           # Dependencies
│
├── archive/                   # Emotion dataset
│   ├── train/
│   └── test/
│
├── saved_faces/               # Face database
│
├── models/
│   ├── emotion_model.pth
│   ├── face_model.pkl
│   └── face_recognizer_model.yml
```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.7+
* Webcam
* Windows / Linux / macOS

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Launch Application

```bash
python app.py
```

### Add New Face

1. Enter person’s name
2. Click **Add Face**
3. System captures 50 face samples automatically

### Start Recognition

* Displays:

  * Person Name
  * Emotion
  * Face Direction
  * Glasses Status

---

## 🧪 Model Overview

### Emotion Recognition CNN

* Input: 48×48 grayscale image
* 3 Convolutional layers
* ReLU + MaxPooling
* Fully connected layers with Dropout
* Output: 7 emotion classes

### Face Recognition

* Haar Cascade face detection
* LBPH feature extraction
* SVM-based classification

---

## 📊 Dataset

### Emotion Dataset

* FER2013
* ~28,000 training images
* 7 emotion classes
* 48×48 grayscale images

### Face Dataset

* Custom dataset created in real time
* 50 samples per person
* Stored as 100×100 grayscale images

---

## 🔮 Future Enhancements

* Age & Gender Detection
* Facial Landmark Detection
* GPU acceleration (CUDA)
* Web-based version (Flask/Django)
* Mobile application support
* Transfer learning for emotion recognition

---

## 📄 License

This project is licensed under the **MIT License**.

---





