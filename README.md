# Pakistani Politician Face Recognition Using Transfer Learning

## Project Overview
This project builds a **face recognition system** that can identify **16 Pakistani politicians** from images using deep learning. Two pre-trained CNN models were fine-tuned using transfer learning: **ResNet50** and **EfficientNet-B2**. Faces were automatically detected and cropped using the **MTCNN** face detector before training.

---

## Group Members
| Name | GitHub |
|------|--------|
| [Salman Shafique] |
| [Shahzad Ahmad] 
|

---

## Models Used
| Model | Test Accuracy | Macro F1 | Weighted F1 |
|-------|--------------|----------|-------------|
| ResNet50 | **83.11%** | 0.83 | 0.84 |
| EfficientNet-B2 | 79.05% | 0.77 | 0.79 |

✅ **ResNet50 performed better** on this dataset.

---

## Politicians (16 Classes)
| # | Name |
|---|------|
| 1 | Ahsan Iqbal |
| 2 | Asif Ali Zardari |
| 3 | Asim Muneer |
| 4 | Benazir Bhutto |
| 5 | Bilawal Bhutto Zardari |
| 6 | Hamza Shehbaz |
| 7 | Hina Rabbani Khar |
| 8 | Imran Khan |
| 9 | Khawaja Asif |
| 10 | Maryam Nawaz |
| 11 | Murad Ali Shah |
| 12 | Nawaz Sharif |
| 13 | Pervez Khattak |
| 14 | Shah Mehmood Qureshi |
| 15 | Shehbaz Sharif |
| 16 | Siraj ul Haq |

---

## Dataset
- **Total Classes:** 16 politicians
- **Images per Class:** ~80 raw images
- **Total Images:** ~1,280
- **Face Detection:** MTCNN (confidence threshold: 90%)
- **Image Size:** 224 × 224 pixels
- **Split:** 75% Train / 15% Validation / 10% Test
- **Test Samples:** 148 images

---

## Pipeline
1. Collect raw images for each politician (stored in Google Drive)
2. Run **MTCNN** face detector to crop and clean faces
3. Split dataset into train / validation / test sets
4. Apply data augmentation (flip, rotation, color jitter, random erasing)
5. Fine-tune **ResNet50** and **EfficientNet-B2** using transfer learning
6. Evaluate both models on test set
7. Compare confusion matrices and misclassified samples

---

## Training Strategy
| Setting | ResNet50 | EfficientNet-B2 |
|---------|----------|-----------------|
| Optimizer | AdamW | AdamW |
| Learning Rate | 2e-4 | 1e-4 |
| Weight Decay | 1e-4 | 1e-4 |
| Loss Function | Cross-Entropy + Label Smoothing (0.05) | Same |
| Max Epochs | 60 | 60 |
| Early Stopping | Patience = 8 | Patience = 8 |
| Stopped at Epoch | 23 | 22 |
| Best Val Accuracy | 83.5% | 81.0% |

---

## Tech Stack
- Python
- PyTorch & Torchvision
- MTCNN
- OpenCV
- scikit-learn
- matplotlib, seaborn
- Google Colab (GPU)

---

## How to Run
1. Open `Project2.ipynb` in Google Colab
2. Mount your Google Drive and place raw images at:
3. /content/drive/MyDrive/raw_dataset/<politician_name>/
4. 3. Install dependencies:
   4. pip install torch torchvision mtcnn opencv-python scikit-learn seaborn tqdm
   5. 4. Run all cells from top to bottom
5. Final cells will print accuracy, confusion matrix, and misclassified images
