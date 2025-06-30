# Part 2: Real-time Violence Detection using PoseLSTM and PoseTransformer

In this phase of the project, we developed a **real-time violence detection pipeline** based on human pose estimation. Our custom architecture, **PoseLSTM** and **PoseTransformer**, utilizes pose landmarks to identify violent actions from webcam or video input in real time.

---

## 🎯 Key Objectives

- Generate a **custom pose-based dataset** using MediaPipe.
- Train an LSTM-based deep learning model (`PoseLSTM`) or Transformer-based deep learning model (`PoseTransformer`) on pose features.
- Deploy a real-time pipeline with **YOLO**, **DeepSORT**, **MediaPipe**, and the trained model.

---

## 📁 Dataset Generation (Custom)

We created our own dataset using the `pose_data_generation.py` script:

1. **Pose Landmark Extraction**
   - Extracted **33 pose landmarks** per person using **MediaPipe Pose**.
   - Each landmark contains: `[x, y, z, visibility]` → Total of **132 features per frame**.

2. **Batching and Labeling**
   - Each sample consists of a fixed number of frames (e.g., 20) forming one batch.
   - Labels are assigned as `Violent` or `Non-Violent` during data generation.

3. **Storage Format**
   - Data is stored as `.csv` files.
   - Each row represents a frame's 132 pose features.

 Custom parameters:
- Number of batches
- Frames per batch
- Label for the sample

---

##  Model Architecture – PoseLSTM

- The model is a **4-layer LSTM** network trained on the `.csv` pose data.
- Trained using the `training_lstm.ipynb` notebook.
- Achieved **100% accuracy** on our generated dataset during evaluation.

**Input shape:** `(batch_size, sequence_length, 132)`  
**Output:** Binary classification (Violent / Non-Violent)

---

## 🎥 Real-time Detection Pipeline

### 🔧 Modules Used:

| Step | Module Used | Purpose |
|------|-------------|---------|
| 1️⃣   | YOLOv8       | Detect humans in frames |
| 2️⃣   | DeepSORT     | Track individual humans |
| 3️⃣   | MediaPipe Pose | Extract 33 pose landmarks/person |
| 4️⃣   | Pose Buffer  | Accumulate 20-frame sequence/person |
| 5️⃣   | PoseLSTM     | Predict violence status per person |

---

## 🛠️ Setup & Run

### 📥 Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name/Website
