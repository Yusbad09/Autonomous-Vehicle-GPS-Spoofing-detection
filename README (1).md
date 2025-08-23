## 🧑‍💻 Author

Developed by **Yusuf Akinkunmi Badrudeen**  
GitHub: (https://github.com/yusbad09)

# 🚗 Autonomous Vehicle GPS Spoofing Detection

This project implements a **GPS Spoofing Detection Pipeline** for autonomous vehicles, with:

- 📡 **Synthetic GNSS + IMU dataset generator** (with spoofed attacks injected)
- 🔬 **Feature engineering** (displacement residuals, rolling stats, heading changes, signal quality)
- 🌲 **RandomForest** and ⚡ **XGBoost** classifiers (with stratified k-fold cross-validation)
- 🧠 Optional **LSTM Autoencoder** anomaly detector
- 📊 Automatic **plots** (confusion matrix, ROC curves, metric bars)
- 📑 End-to-end **PDF report** generation (via [reportlab](https://www.reportlab.com/))

---

## 📂 Project Structure

```
autonomous-vehicle-gps-spoofing-detection/
│── pipeline.ipynb        # Jupyter notebook (step-by-step tutorial)
│── outputs_cv/           # Saved metrics, plots, report.pdf
│── README.md             # Documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/autonomous-vehicle-gps-spoofing-detection.git
cd autonomous-vehicle-gps-spoofing-detection

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** should contain:
```
numpy
pandas
matplotlib
scikit-learn
xgboost
torch
reportlab
```

---

## 🛰️ Pipeline Overview

1. **Synthetic Dataset Generation**  
   - True trajectory simulated as random walk in latitude/longitude.  
   - IMU data (speed, acceleration, gyro) derived.  
   - GNSS readings perturbed with noise + spoofed windows (abrupt or gradual).  

   Example spoof windows:
   ```python
   spoof_windows=[(400,480), (1100,1180), (1600,1680)]
   ```

2. **Feature Engineering**  
   - GNSS displacement vs inertial displacement.  
   - Residual errors and rolling means/std.  
   - Heading change, SNR, satellite count.  

3. **Cross-Validation (Supervised Models)**  
   - RandomForest (200 trees).  
   - XGBoost (200 estimators).  
   - Metrics: Accuracy, Precision, Recall, F1, ROC AUC.  

   **Plots saved per fold**:
   - Confusion matrix  
   - ROC curve  
   - Metric bar chart  

4. **LSTM Autoencoder (Optional)**  
   - Trained only on **clean (non-spoofed)** samples.  
   - Detects anomalies by high reconstruction error.  
   - Threshold = 95th percentile of clean errors.  

5. **PDF Report**  
   - Summarizes per-fold + average metrics.  
   - Embeds plots (confusion, ROC, AE loss/error).  
   - Autoencoder threshold + metrics included.  

---

## 📊 Example Results

### Cross Validation (5 folds)

| Model         | Accuracy | Precision | Recall | F1   | ROC AUC |
|---------------|----------|-----------|--------|------|---------|
| RandomForest  | 0.9950   | 0.992     | 0.967  | 0.979| 0.9997  |
| XGBoost       | 0.9985   | 1.000     | 0.988  | 0.994| 0.9989  |

### Autoencoder (Unsupervised)

| Accuracy | Precision | Recall | F1   | ROC AUC |
|----------|-----------|--------|------|---------|
| 0.9245   | 0.668     | 0.738  | 0.701| 0.914   |

📑 Full PDF Report (`outputs_cv/report.pdf`) is automatically generated with tables + figures.

---

## ▶️ Usage

Run the pipeline via Jupyter Notebook:

```bash
jupyter notebook gps_spoofing_detection_pipeline.ipynb
```

Or directly from Python:

```bash
python pipeline.py
```

Results will be saved under `outputs_cv/`.

---


---

## 🔮 Extensions

- Test with **real GNSS/IMU logs** (replace the synthetic generator).  
- Add more models (LightGBM, CatBoost).  
- Try temporal sequence models (RNNs, Transformers) for supervised detection.  
- Multi-class spoofing (clean / abrupt / gradual).  

---


