# 🎭 Face and Emotion Recognition System

A comprehensive real-time face recognition and emotion detection system built with Python, OpenCV, and PyTorch. This project combines computer vision, machine learning, and deep learning technologies to identify faces and analyze emotions simultaneously.

## 🌟 Features Overview

### 🔍 Face Recognition
- **Multi-person Face Database**: Store and recognize multiple individuals
- **Real-time Face Detection**: Live webcam face detection using Haar Cascades
- **LBPH Face Recognition**: Local Binary Pattern Histogram for face identification
- **SVM Classification**: Support Vector Machine for improved accuracy
- **Automatic Sample Collection**: Collects 15 samples per person for training
- **Model Persistence**: Save and load trained models automatically
- **Confidence Scoring**: Reliability metrics for recognition results

### 😊 Emotion Recognition
- **7 Emotion Categories**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Custom CNN Architecture**: 3-layer convolutional neural network
- **PyTorch Implementation**: Deep learning framework for emotion classification
- **Real-time Processing**: Live emotion detection from webcam feed
- **Confidence Thresholding**: Filters low-confidence predictions
- **Preprocessing Pipeline**: Image normalization and feature extraction

### 🎯 Advanced Detection Features
- **Face Direction Tracking**: Detects if person is looking Left/Right/Forward
- **Eye Tracking**: Eye position analysis for direction detection
- **Glasses Detection**: Automatic detection of eyewear
- **Multi-face Processing**: Handles multiple faces in single frame
- **Adaptive Thresholding**: Dynamic confidence adjustment

### 💻 User Interface
- **Tkinter GUI**: User-friendly graphical interface
- **Live Camera Feed**: Real-time video display (640x480)
- **Interactive Controls**: Add faces, start/stop recognition
- **Status Indicators**: Progress tracking and system feedback
- **Results Display**: Live emotion, direction, and recognition results

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.7+**: Primary programming language
- **OpenCV 4.5+**: Computer vision and image processing
- **PyTorch 2.0+**: Deep learning framework
- **NumPy**: Numerical computations and array operations
- **Tkinter**: GUI framework for desktop application

### Machine Learning Libraries
- **scikit-learn**: Machine learning utilities and algorithms
- **Pillow (PIL)**: Image processing for GUI display
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization
- **Pandas**: Data manipulation and analysis

### Computer Vision Components
- **Haar Cascade Classifiers**: Face and eye detection
- **LBPH (Local Binary Pattern Histogram)**: Face recognition algorithm
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Histogram Equalization**: Image preprocessing
- **Gaussian Filtering**: Noise reduction

### Deep Learning Architecture
- **Custom CNN**: 3-layer convolutional neural network
- **PyTorch DataLoader**: Efficient data loading and batching
- **Adam Optimizer**: Adaptive learning rate optimization
- **Cross-Entropy Loss**: Multi-class classification loss function
- **Dropout Regularization**: Prevents overfitting

## 📁 Project Structure

```
ml_project/
│
├── 📄 app.py                          # Main GUI application
├── 📄 emotion_recognition.py          # Emotion detection system
├── 📄 face_recognition.py            # Face recognition system
├── 📄 train.py                       # Model training scripts
├── 📄 face_detection.py              # Face detection utilities
├── 📄 download_samples.py            # Sample data downloader
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Project documentation
│
├── 📁 archive/                       # Emotion training dataset
│   ├── 📁 train/                     # Training images
│   │   ├── 📁 angry/                 # Angry emotion samples
│   │   ├── 📁 disgust/               # Disgust emotion samples
│   │   ├── 📁 fear/                  # Fear emotion samples
│   │   ├── 📁 happy/                 # Happy emotion samples
│   │   ├── 📁 neutral/               # Neutral emotion samples
│   │   ├── 📁 sad/                   # Sad emotion samples
│   │   └── 📁 surprise/              # Surprise emotion samples
│   └── 📁 test/                      # Test images (same structure)
│
├── 📁 saved_faces/                   # Face recognition database
│   ├── 📁 person1/                   # Individual person folders
│   ├── 📁 person2/                   # Face samples per person
│   └── 📁 [name]/                    # Dynamic person directories
│
├── 📁 dataset/                       # Additional face datasets
├── 📁 __pycache__/                   # Python bytecode cache
│
├── 🤖 emotion_model.pth              # Trained emotion CNN model
├── 🤖 emotion_detection_model.pb     # TensorFlow model (if used)
├── 🤖 face_recognizer_model.yml      # OpenCV face recognizer
├── 🤖 face_recognition_data.pkl      # Pickled recognition data
├── 🤖 face_model.pkl                 # SVM face classification model
│
└── 📊 confusion_matrix.png           # Model evaluation results
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam/Camera device
- Windows/Linux/macOS

### 1. Clone Repository
```bash
git clone <repository-url>
cd ml_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages:
```bash
pip install opencv-python>=4.5.0
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.19.0
pip install scikit-learn>=0.24.0
pip install Pillow>=10.2.0
pip install matplotlib>=3.7.2
pip install seaborn>=0.12.2
pip install pandas>=1.2.0
pip install requests>=2.31.0
pip install imutils
```

### 3. Download Sample Data (Optional)
```bash
python download_samples.py
```

## 🎮 Usage Guide

### 🖥️ Launch GUI Application
```bash
python app.py
```

### 📝 Adding New Faces
1. Enter person's name in the text field
2. Click "Add Face" button
3. Position face in camera frame
4. System automatically collects 15 samples
5. Model retrains automatically

### 🎭 Real-time Recognition
1. Click "Start Recognition" button
2. Face recognition and emotion detection run simultaneously
3. Results display: Name, Emotion, Confidence, Direction, Glasses
4. Click "Stop Recognition" to end session

### 💾 Model Management
- Click "Save Model" to persist training data
- Models auto-load on application startup
- Training data stored in `saved_faces/` directory

### 🧪 Direct Emotion Recognition
```bash
python emotion_recognition.py
```

### 📊 Train Custom Models
```bash
python train.py --dataset ./archive/train --model face_model.pkl
```

## 🔧 Technical Specifications

### CNN Architecture (Emotion Recognition)
```python
Input: 48x48 grayscale images
├── Conv2D(1→32, kernel=3x3, padding=1)
├── ReLU + MaxPool2D(2x2)
├── Conv2D(32→64, kernel=3x3, padding=1)
├── ReLU + MaxPool2D(2x2)
├── Conv2D(64→128, kernel=3x3, padding=1)
├── ReLU + MaxPool2D(2x2)
├── Flatten → 128*6*6 = 4608 features
├── Linear(4608→64) + ReLU
├── Dropout(0.5)
└── Linear(64→7) → Softmax
```

### Face Recognition Pipeline
```python
Input: BGR image
├── Haar Cascade Face Detection
├── Face Cropping & Resizing (100x100)
├── Grayscale Conversion
├── Histogram Equalization
├── LBPH Feature Extraction
├── SVM Classification
└── Confidence Scoring
```

### Performance Metrics
- **Face Detection**: Haar Cascades with 1.1 scale factor
- **Emotion Recognition**: 48x48 input, ~10 FPS processing
- **Face Recognition**: 100x100 input, 15 samples per person
- **Confidence Threshold**: 20% for emotions, 30% for faces
- **GUI Resolution**: 640x480 camera display

## 🎯 Key Algorithms

### Face Recognition (LBPH)
- **Local Binary Patterns**: Texture feature extraction
- **Histogram Comparison**: Statistical face matching
- **Multi-scale Detection**: Various face sizes support
- **Label Encoding**: Efficient name-to-number mapping

### Emotion Recognition (CNN)
- **Convolutional Layers**: Feature extraction from facial expressions
- **Max Pooling**: Spatial dimension reduction
- **Fully Connected**: High-level feature classification
- **Dropout Regularization**: Overfitting prevention

### Direction Detection
- **Eye Tracking**: Pupil position analysis
- **Face Positioning**: Relative position in frame
- **Geometric Analysis**: Eye distance calculations
- **Fallback Logic**: Multiple detection methods

## 📊 Dataset Information

### Emotion Dataset Structure
- **7 Emotion Classes**: Balanced distribution
- **Training Set**: ~28,000+ images
- **Test Set**: ~7,000+ images
- **Image Format**: 48x48 grayscale
- **Data Augmentation**: Rotation, scaling, flipping

### Face Recognition Database
- **Dynamic Addition**: Real-time face enrollment
- **Sample Collection**: 15 images per person
- **Storage Format**: 100x100 grayscale
- **Preprocessing**: Histogram equalization, noise reduction

## 🔍 Features Deep Dive

### 1. Real-time Face Detection
- Uses OpenCV's Haar Cascade classifiers
- Detects faces in live video stream
- Minimum face size: 100x100 pixels
- Multiple face support in single frame

### 2. Emotion Classification
- **7 Emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Deep Learning**: Custom CNN with 3 convolutional layers
- **Preprocessing**: Resizing to 48x48, normalization
- **Confidence**: Softmax probability distribution

### 3. Face Direction Analysis
- **Eye Tracking**: Detects eye positions for gaze estimation
- **3 Directions**: Left, Right, Forward
- **Fallback**: Face position analysis when eyes not detected
- **Sensitivity**: Adjustable thresholds for direction detection

### 4. Advanced Features
- **Glasses Detection**: Identifies if person wears eyewear
- **Multi-person**: Processes multiple faces simultaneously
- **Adaptive**: Dynamic confidence thresholds
- **Real-time**: 10+ FPS processing speed

## 🛡️ Error Handling

### Robust Error Management
- **Camera Access**: Handles camera initialization failures
- **Model Loading**: Graceful degradation for missing models
- **Face Detection**: Continues operation when no faces detected
- **File I/O**: Safe file operations with exception handling

### Fallback Mechanisms
- **Emotion Recognition**: Returns "neutral" for low confidence
- **Face Recognition**: Returns "Unknown" for unrecognized faces
- **Direction Detection**: Multiple detection algorithms
- **GUI**: Maintains responsive interface during errors

## 🔮 Future Enhancements

### Planned Features
- [ ] **Age Estimation**: Deep learning age prediction
- [ ] **Gender Recognition**: Binary classification for gender
- [ ] **Facial Landmarks**: 68-point facial feature detection
- [ ] **3D Face Analysis**: Depth-based face recognition
- [ ] **Voice Recognition**: Audio-visual fusion
- [ ] **Database Integration**: MySQL/PostgreSQL support
- [ ] **Web Interface**: Flask/Django web application
- [ ] **Mobile App**: Android/iOS mobile versions

### Technical Improvements
- [ ] **GPU Acceleration**: CUDA support for faster processing
- [ ] **Model Optimization**: TensorRT optimization
- [ ] **Data Augmentation**: Advanced image transformations
- [ ] **Transfer Learning**: Pre-trained model fine-tuning
- [ ] **Ensemble Methods**: Multiple model combination
- [ ] **Real-time Tracking**: Face tracking across frames

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

### Project Maintainer
- **Name**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [your-username]

### Issues & Bug Reports
- Create issues on GitHub repository
- Include error logs and system information
- Provide steps to reproduce problems

### Feature Requests
- Submit feature requests via GitHub issues
- Describe use case and expected behavior
- Include mockups or examples if applicable

## 🙏 Acknowledgments

### Technologies & Libraries
- **OpenCV Team**: Computer vision library
- **PyTorch Team**: Deep learning framework
- **scikit-learn**: Machine learning library
- **Python Community**: Programming language and ecosystem

### Datasets
- **FER2013**: Facial Expression Recognition dataset
- **LFW**: Labeled Faces in the Wild dataset
- **CelebA**: Celebrity faces dataset

### Research & Inspiration
- Face recognition research papers
- Emotion recognition studies
- Computer vision conferences (CVPR, ICCV, ECCV)

---

⭐ **Star this repository if you found it helpful!** ⭐

🔄 **Keep checking for updates and new features!** 🔄 #   F a c e - E m o t i o n - R e c o g n i t i o n - S y s t e m - U s i n g - N e u r a l - N e t w o r k s  
 