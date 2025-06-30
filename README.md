#  Real-time Violence Detection using PoseLSTM and PoseTransformer

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
- Trained using the `poselstm.ipynb` and `posetransformer.ipynb` notebook.
- Achieved **99.95% accuracy** on **PoseLSTM** and **98.59% accuracy** on **PoseTransformer** models using our generated dataset during evaluation.

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
| 5️⃣   | PoseModel     | Predict violence status per person |

---

## 🛠️ Setup & Run

```bash

# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git

# Install Requirements
pip install -r requirements.txt

# Change directory to Website
cd Website

# Run the Flask app
python main.py
```

---

## 👥 Authors

- **Soumava Das** – VioLENS: Intelligent Violence Detection System (2025)
- **Sarthak Saha** – VioLENS: Intelligent Violence Detection System (2025)
- **Debshankar Dey** – VioLENS: Intelligent Violence Detection System (2025)
- **Avipriya Ghosh** – VioLENS: Intelligent Violence Detection System (2025)

---

## Acknowledgement

- We would like to thank N.Wokje for providing the deep_sort code at [Deep Sort](https://github.com/nwojke/deep_sort)
- We would also like to thank Model Bunker for providing the deep_sort code implemened using pytorch that helped in our project [Deepsort: using Pytorch](https://github.com/ModelBunker/Deep-SORT-PyTorch)

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0) due to the usage of [DeepSort](https://github.com/ModelBunker/DeepSort-YOLO), which is GPL-licensed.

See the [LICENSE](./LICENSE) file for full details.
