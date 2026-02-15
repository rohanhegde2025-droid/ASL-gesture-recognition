# ASL-gesture-recognition

A real-time American Sign Language (ASL) gesture recognition system using MediaPipe hand landmarks and a Random Forest classifier. The system detects hand gestures through a webcam and predicts ASL letters in real time.

---

## Demo

> Show letter signs in front of your webcam and the system will predict the ASL letter in real time with a confidence score.

---

<img width="978" height="754" alt="image" src="https://github.com/user-attachments/assets/1eccd6f6-3575-461c-9f3a-74f6182cd729" />
<img width="1600" height="890" alt="image" src="https://github.com/user-attachments/assets/28d734ce-d2ef-4c5a-87cc-5509153937a7" />
<img width="1600" height="896" alt="image" src="https://github.com/user-attachments/assets/6068f435-7097-47f6-857b-288a75664806" />
<img width="1600" height="926" alt="image" src="https://github.com/user-attachments/assets/83b7e3b8-9e92-47a7-9f17-e4d19d5e4233" />





## How It Works

```
Webcam Frame → MediaPipe (extracts 21 hand landmarks) → 63 numbers → Random Forest → Predicted Letter
```

Instead of training on raw pixels, the system uses **MediaPipe** to extract 21 hand landmark coordinates (x, y, z) per frame — giving the model a structured, background-invariant representation of the hand. This makes training fast and inference robust across different cameras and lighting conditions.

---

## Project Structure

```
asl-gesture-recognition/
├── asl-dataset/          # Dataset folder (not tracked by git)
│   ├── asl_dataset_small/  # Sampled dataset (700 images per class)
│   └── landmarks.csv       # Extracted landmarks (generated)
├── models/               # Saved model files (not tracked by git)
│   ├── asl_classifier.pkl
│   └── label_encoder.pkl
├── src/
│   ├── extract_landmarks.py  # MediaPipe landmark extraction
│   ├── train.py              # Model training script
│   └── inference.py          # Real-time webcam inference
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam capture and display |
| scikit-learn | Random Forest classifier |
| NumPy & Pandas | Data handling |
| Python 3.10 | Runtime |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/asl-gesture-recognition.git
cd asl-gesture-recognition
```

### 2. Create a virtual environment with Python 3.10
```bash
py -3.10 -m venv asl_env
```

### 3. Activate the virtual environment

On Windows (PowerShell):
```bash
asl_env\Scripts\activate
```

On Windows (Git Bash):
```bash
source asl_env/Scripts/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle — 87,000 images across 29 classes (A–Z + `del`, `space`, `nothing`).

For faster training, only 700 images per class are sampled using `smaller_samples.py`.

Download the dataset from Kaggle and place it inside:
```
asl-dataset/asl_alphabet_train/
```

---

## Usage

### Step 1 — Sample the dataset (700 images per class)
```bash
python smaller_samples.py
```

### Step 2 — Extract hand landmarks
```bash
python src/extract_landmarks.py
```
This runs MediaPipe on every image and saves a `landmarks.csv` file with 63 normalized landmark values per image.

### Step 3 — Train the model
```bash
python src/train.py
```
Trains a Random Forest classifier on the extracted landmarks. Typical accuracy: **93–97%** on test data. Model is saved to `models/`.

### Step 4 — Run real-time webcam inference
```bash
python src/inference.py
```
Opens your webcam. Show an ASL letter sign and the predicted letter appears on screen with a confidence score. Press `Q` to quit.

---

## Classes Supported

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z + del + space + nothing
```

> Note: J and Z involve motion gestures and may have lower accuracy with a static landmark approach.

---

## Key Design Decisions

**Why MediaPipe landmarks instead of raw images?**
Training a CNN on raw images requires large datasets, long training times, and struggles to generalize across different cameras and backgrounds. MediaPipe extracts structured hand geometry (where each finger joint is) which is camera and background agnostic — making training fast and inference robust.

**Why Random Forest?**
For a 63-feature input space with 29 classes, Random Forest trains in under 2 minutes on CPU and achieves comparable accuracy to a small neural network. It's also interpretable and doesn't require GPU resources.

**Why wrist normalization?**
Landmark coordinates are normalized relative to the wrist (landmark 0) before training. This makes the model invariant to hand distance from the camera — a hand close to or far from the camera produces the same feature vector.

---

## Results

| Metric | Value |
|---|---|
| Training samples | ~15,000 |
| Test accuracy | 93–97% |
| Training time | < 2 minutes |
| Inference speed | Real-time (30 FPS) |

---

## Limitations

- J and Z require motion and are harder to classify with static landmarks
- Performance may vary under poor lighting conditions
- Currently supports single hand detection only

---

## Future Improvements

- Add support for dynamic gestures (J, Z) using sequence models (LSTM)
- Build a word accumulator that strings letters into words
- Collect personal webcam data to fine-tune for individual hand shapes
- Add a Keras MLP for potentially higher accuracy

---
