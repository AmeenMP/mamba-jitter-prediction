# Optimizing and Benchmarking Mamba for Network Traffic Prediction on Edge Devices

This repository accompanies the Master's thesis:

**"Optimizing and Benchmarking MAMBA for Network Traffic Prediction on Edge Devices"**  
_Ameen Maniparambath, University of Oulu, June 2025_

---

## 📘 Overview

This project explores the use of the Mamba architecture—a recent advancement in state space models—for forecasting jitter in LTE network traffic. It evaluates Mamba’s suitability for edge deployment by benchmarking its prediction accuracy, computational efficiency, and runtime behavior against conventional deep learning models, with a special focus on the NVIDIA Jetson Orin Nano platform.

---

## 🎯 Objectives

- Forecast network jitter using the Mamba model
- Benchmark performance against CNN-LSTM baseline
- Evaluate efficiency, inference speed, and memory usage
- Attempt edge deployment and document challenges

---

## 📁 Repository Structure

mamba-jitter-prediction/
├── src/
│ ├── train.py # Mamba training script
│ └── evaluate.py # Evaluation and prediction visualization
├── plots/
│ ├── training_loss.png # Training/Validation loss over epochs
│ ├── validation_prediction.png # Ground truth vs. predicted (validation set)
│ └── test_prediction.png # Ground truth vs. predicted (test set)
├── requirements.txt # Cleaned list of required Python packages
├── README.md # Project documentation



---

## ⚙️ Setup & Installation

### 1. Clone the repository:
```bash
git clone https://github.com/AmeenMP/mamba-jitter-prediction.git
cd mamba-jitter-prediction



2. Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt



🧪 Usage
Train the model:
python src/train.py


Evaluate the model:
python src/evaluate.py



📊 Visualizations
The plots/ folder includes key evaluation plots:

training_loss.png: Training and validation loss progression

validation_prediction.png: Model predictions vs actual jitter (validation)

test_prediction.png: Model predictions vs actual jitter (test)


📂 Dataset
This project uses LTE network traffic data, including jitter measurements and CQI values, from three user devices (UE1–UE3). The final model was trained on UE3 data.



 Dataset Access Notice:
The dataset is not openly available for public download. It must be requested via the official IEEE DataPort page below.
The authors of this thesis do not have permission to redistribute the dataset.

Dataset Title: UE Statistics Time-Series (CQI) in LTE Networks
Access Link: https://ieee-dataport.org/documents/ue-statistics-time-series-cqi-lte-networks
Citation:I. Chatzistefanidis, N. Makris, V. Passas, and T. Korakis, 
“UE statistics time-series (CQI) in LTE networks,” 2022. 
[Online]. Available: https://dx.doi.org/10.21227/ec7p-xq38



🧠 Model Overview
Model: Mamba (structured state space model)

Input: Sequences of 20 timesteps with 21 features

Output: One-step-ahead jitter forecast for UE3

Loss: Smooth L1 Loss

Optimizer: AdamW

Scheduler: Cosine Annealing

Baseline Comparison: CNN-LSTM (Ijlal Khan, 2022)

Although deployment to Jetson Orin Nano was attempted, it was not successful due to ARM64 incompatibility with the Triton backend used in Mamba.



🚫 Limitations
Dataset is private and must be requested from IEEE DataPort.

Model was not successfully deployed on Jetson hardware due to runtime library limitations (Triton / ARM64).


📜 License
This project is provided for academic reference. If you use this work, please cite the thesis and the dataset authors appropriately.


## 📫 Contact

This repository was created as part of a Master's thesis at the University of Oulu.

For academic inquiries or collaboration:
**Ameen Maniparambath**  
GitHub: [AmeenMP](https://github.com/AmeenMP)





