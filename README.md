# 🎭 AMCEF: Adaptive Multi-modal Confidence-based Emotion Fusion System

## Overview

AMCEF (Adaptive Multi-modal Confidence-based Emotion Fusion System) is an Audio–Visual Emotion Recognition framework that combines speech-based and facial-expression-based emotion analysis to generate reliable and robust emotion predictions.

The system integrates three independently trained Speech Emotion Recognition (SER) models trained on different emotional speech datasets and combines their outputs with video-based emotion predictions through an adaptive confidence-based fusion mechanism.

By leveraging both audio and visual information, AMCEF improves emotion recognition performance compared to traditional single-modality approaches.

---

## Features

### Audio Emotion Recognition
- MFCC-based feature extraction
- Multi-dataset speech emotion recognition
- Ensemble of three trained models:
  - EMO-DB (German Emotional Speech Database)
  - EMOVO (Italian Emotional Speech Database)
  - SHEMO (Persian Emotional Speech Database)

### Video Emotion Recognition
- Facial emotion analysis from video input
- Support for:
  - Uploaded video files
  - Live camera input
  - Simulated video predictions using reference CSV data

### Adaptive Fusion Engine
- Confidence-based multimodal fusion
- Dynamic fusion strategy selection
- Emotion reliability assessment

### Analytics and Visualization
- Emotion Reliability Index (ERI)
- Emotion Stability Score (ESS)
- Agreement Score Analysis
- Emotion Probability Distribution
- Fusion Decision Explanation

### Interactive Dashboard
- Streamlit-based user interface
- Real-time emotion visualization
- Developer debug panel
- Emotion confidence monitoring

---

## System Architecture

```text
Audio Input
      │
      ▼
MFCC Feature Extraction
      │
      ▼
 ┌─────────────┬─────────────┬─────────────┐
 │             │             │
 ▼             ▼             ▼
EMO-DB      EMOVO        SHEMO
Model       Model        Model
 │             │             │
 └─────────────┼─────────────┘
               ▼
      Audio Emotion Scores

Video Input
      │
      ▼
Video Processing
      │
      ▼
Video Emotion Scores

               ▼
      Adaptive Fusion Engine
               │
               ▼
     Final Emotion Prediction
               │
               ▼
      ERI, ESS & Analytics
```

---

## Project Structure

```text
AMCEF/
│
├── backend/
│   ├── app.py
│   ├── fusion_predict.py
│   └── feature_extraction.py
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── emodb_model.keras
│   ├── emovo_model.keras
│   └── shemo_model.keras
│
├── assets/
│   ├── video_results_100.csv
│   └── sample_data/
│
├── evaluation/
│   └── evaluate.py
│
├── requirements.txt
├── walkthrough.md
└── README.md
```

---

## Methodology

### Step 1: Input Acquisition

The user provides:

- Audio file (.wav)
- Video file (.mp4, .flv, .avi)
- Live microphone input
- Live camera input

---

### Step 2: Audio Feature Extraction

Audio signals are processed using Mel-Frequency Cepstral Coefficients (MFCC).

Process:

1. Audio Loading
2. Resampling to 16kHz
3. MFCC Extraction
4. Statistical Aggregation
5. Feature Normalization

---

### Step 3: Audio Emotion Prediction

The extracted MFCC features are passed through:

- EMO-DB Model
- EMOVO Model
- SHEMO Model

Each model generates:

- Emotion Label
- Confidence Score
- Probability Distribution

---

### Step 4: Video Emotion Prediction

#### Demo Mode
Uses:

```text
video_results_100.csv
```

to simulate emotion predictions.

#### Real-Time Mode

1. Face Detection using OpenCV
2. Frame Extraction
3. Facial Expression Recognition
4. Probability Aggregation

---

### Step 5: Adaptive Fusion

Audio and video predictions are fused using confidence-aware decision logic.

Fusion strategies include:

- Agreement Boost
- Audio Dominance
- Audio Priority
- Weighted Fusion

---

### Step 6: Reliability Analysis

The system computes:

#### Emotion Reliability Index (ERI)

Measures prediction reliability.

| ERI Score | Reliability |
|------------|------------|
| 0.80 – 1.00 | High |
| 0.50 – 0.79 | Medium |
| Below 0.50 | Low |

#### Emotion Stability Score (ESS)

Measures temporal consistency of detected emotions.

| ESS Score | Stability |
|------------|------------|
| 0.80 – 1.00 | High |
| 0.50 – 0.79 | Moderate |
| Below 0.50 | Low |

---

## Model Performance

| Dataset | Validation Accuracy |
|----------|--------------------|
| EMO-DB | 83% |
| EMOVO | 75% |
| SHEMO | 95% |

### Audio Fusion Weights

| Model | Weight |
|---------|---------|
| EMO-DB | 0.30 |
| EMOVO | 0.20 |
| SHEMO | 0.50 |

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd AMCEF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Backend

```bash
cd backend
uvicorn app:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

## Running the Frontend

```bash
cd frontend
streamlit run app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## Dashboard Features

- Emotion Prediction Card
- Audio Emotion Analysis
- Video Emotion Analysis
- Fusion Intelligence
- Emotion Reliability Index (ERI)
- Agreement Score Monitor
- Emotion Stability Monitor (ESS)
- Probability Distribution Charts
- Fusion Strategy Intelligence
- Developer Debug Panel

---

## Applications

### Mental Health Monitoring
Assist healthcare professionals in understanding emotional patterns.

### Online Learning Platforms
Monitor student engagement and emotional responses.

### Customer Experience Analytics
Analyze customer emotions during service interactions.

### Human–Computer Interaction
Enable emotionally aware intelligent systems.

### Recruitment and Interview Analysis
Support communication and behavioral assessment.

### Healthcare Support Systems
Assist in patient emotion monitoring and wellbeing analysis.

---

## Future Enhancements

- Real-time FER-CNN integration
- Transformer-based audio emotion models
- Explainable AI visualizations
- Multilingual emotion recognition
- Edge deployment support
- Cross-cultural emotion adaptation

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- OpenCV
- FastAPI
- Streamlit
- Scikit-learn

---

## Academic Information

**Project Title:**  
Adaptive Multi-modal Confidence-based Emotion Fusion System (AMCEF)

**Domain:**  
Artificial Intelligence and Data Science

**Category:**  
Audio–Visual Emotion Recognition

---

## License

This project is developed for academic and research purposes only.

---

**AMCEF – Enhancing Emotion Recognition through Adaptive Audio–Visual Fusion Intelligence**