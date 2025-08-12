# Autonomous-Vehicle-GPS-Spoofing-detection
Synthetic GPS + IMU Spoofing Detection using Machine Learning
Overview
This project demonstrates a machine learning pipeline for detecting GPS spoofing attacks using synthetic GPS and IMU (Inertial Measurement Unit) data.
The notebook is fully documented and designed for both educational purposes and as a research baseline for further studies on GNSS security in autonomous systems, drones, and IoT devices.

The system compares XGBoost and Random Forest classifiers for accuracy and performance in spoofing detection.

**Dataset**
Since real-world spoofing datasets are limited and often restricted for security reasons, this project uses a synthetic dataset generator that simulates:

GPS coordinates with realistic noise

IMU readings (acceleration, gyroscope)

Spoofing attack scenarios (location jumps, inconsistent velocity, IMU mismatch)

Synthetic data ensures fast execution and privacy-safe experimentation while maintaining realistic patterns.

**Notebook Structure**
The notebook is divided into the following main sections:

Setup & Imports

Install dependencies

Import Python libraries

Synthetic Dataset Generation

Generate GPS + IMU readings under normal and spoofing conditions

Data visualization for inspection

Feature Engineering & Preprocessing

Compute derived features (velocity, acceleration mismatches)

Normalize features for model input

Model Training & Evaluation

XGBoost classifier

Random Forest classifier

Accuracy, precision, recall, F1-score comparison

Confusion matrices

Model Comparison

Performance metrics table

Recommendations for further tuning

**Results Summary**
Model	Accuracy	Precision	Recall	F1-Score
XGBoost	~99%	High	High	High
Random Forest	~98%	High	High	High

Both models perform well, but XGBoost shows slightly better results on synthetic data.

**How to Run**
Option 1: Google Colab (Recommended)
Open the .ipynb file in Colab

Run all cells in sequence

Option 2: Local Environment
bash
Copy
Edit
**git clone https://github.com/yusbad09/Autonomous-Vehicle-GPS-Spoofing-detection.git**
cd Autonomous-Vehicle-GPS-Spoofing-detection
pip install -r requirements.txt
jupyter notebook spoofing_detection.ipynb
Possible Extensions
Train on real-world GNSS spoofing datasets

Add deep learning models (e.g., LSTM for time-series detection)

Deploy as a real-time detection system on embedded devices

Integrate with multi-sensor fusion for resilience against spoofing

License
This project is released under the MIT License for academic and research use
