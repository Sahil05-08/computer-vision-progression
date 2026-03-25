# Computer Vision Progression
### From Basic CNN → Transfer Learning → Deployed Web Application

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Real--Time-green)
![ResNet50](https://img.shields.io/badge/ResNet50-97.14%25_Accuracy-brightgreen)

---

## What this project is

A complete computer vision system built in three progressive stages — from a failing basic CNN, to a high-accuracy transfer learning model, to a fully deployed real-time detection system with a web interface.

This is not a tutorial follow-along. Each stage was built to solve a specific problem that the previous stage couldn't.

---

## The problem I was solving

Classify and detect animals and people accurately in real time — with no GPU, on a standard laptop webcam — and deploy the classifier as a usable web application.

---

## Stage 1 — Basic CNN from scratch (diagnosed the failure)

Built a simple convolutional neural network trained on 275 images (cats and dogs).

**Result:** Poor generalization. Loss plateaued early. Model failed on unseen images.

**Root cause:** Too little data for a network learning visual features from random weights.

**Learning:** Small datasets need pretrained visual knowledge — not random initialization.

---

## Stage 2 — Transfer Learning with ResNet50 (fixed accuracy)

Used ResNet50 pretrained on ImageNet (1.2M images, 1000 classes). Froze all base layers to preserve learned visual features. Replaced only the final fully connected layer for 2-class output.

**Results:**
| Metric | Value |
|---|---|
| Validation Accuracy | **94.54%** |
| Training Loss | 0.66 → 0.15 (10 epochs) |
| Total Parameters | 23.5 million |
| Trainable Parameters | 4,098 (last layer only) |
| Training Time | ~5–10 minutes on CPU |
| Input Size | 128 × 128 pixels |

**Learning:** Freezing 23.5M pretrained parameters and retraining only 4,098 achieves 97% accuracy on the same small dataset that the full CNN failed on. Pretrained features transfer directly.

**Tech:** PyTorch · torchvision · ResNet50 · ImageFolder · DataLoader

---

## Stage 3A — Real-Time Object Detection (YOLOv8 + OpenCV)

ResNet50 classifies a single image but cannot localize multiple objects or process live video. Integrated YOLOv8n with OpenCV for real-time webcam detection.

**Features:**
- Multi-object detection across 80+ classes simultaneously
- Confidence filtering at 0.7 threshold — reduces false positives
- Live FPS monitoring and display
- Real-time people counting with crowd state alerts:
  - Empty — no person detected
  - Single person detected
  - Multiple people detected
- Frame resizing to 500×320 for CPU performance optimization

**Performance:**
| Metric | Value |
|---|---|
| FPS on CPU | ~3 FPS |
| Inference speed | ~340ms per frame |
| Confidence threshold | 0.7 |
| GPU required | No |

**Tech:** YOLOv8n · Ultralytics · OpenCV · Python

**Business use case:** Retail footfall monitoring, security systems, classroom attendance — any environment needing live crowd awareness without expensive hardware.

---

## Stage 3B — Streamlit Web Application (deployed classifier)

Built a complete web interface around the ResNet50 model so anyone can use it without writing code.

**App features:**
- Upload any image (JPG, JPEG, PNG, GIF, BMP) and get instant prediction
- Shows predicted class (Cat / Dog) with confidence percentage
- Visual confidence breakdown — probability for each class displayed separately
- Progress bar showing prediction confidence
- One-click sample image testing from the validation dataset
- Model information panel explaining the architecture
- Runs entirely on CPU — no GPU needed
- Auto-detects GPU if available and switches automatically

**How the prediction works:**
1. User uploads image via browser
2. Image resized to 128×128 and converted to tensor
3. Passed through frozen ResNet50 base (23.5M ImageNet features)
4. Final FC layer outputs 2-class logits
5. Softmax converts to probabilities
6. Result and confidence displayed instantly

**Tech:** Streamlit · PyTorch · ResNet50 · PIL · Python

---

## Project files

```
computer-vision-progression/
│
├── Computer_vision_progression.ipynb   # Full training journey (Stages 1 + 2)
├── app.py                              # Streamlit web application (Stage 3B)
├── README.md
└── .gitignore
```

> **Note:** Model weights (`cat_dog_model.pth`) and dataset images are not included due to file size. Run the notebook to retrain the model. YOLOv8 weights download automatically via ultralytics.

---

## How to run

**Install dependencies:**
```bash
pip install torch torchvision streamlit ultralytics opencv-python pillow
```

**Run the web app:**
```bash
streamlit run app.py
```

**Run the notebook (to retrain):**
```bash
jupyter notebook Computer_vision_progression.ipynb
```

**Run real-time detection:**
```bash
python detect.py
```

---

## Key concepts demonstrated

| Concept | Where |
|---|---|
| CNN architecture and training loop | Stage 1 |
| Diagnosing overfitting on small datasets | Stage 1 |
| Transfer learning — freezing base layers | Stage 2 |
| Fine-tuning final FC layer only | Stage 2 |
| Achieving 97% accuracy with 275 images | Stage 2 |
| Real-time multi-object detection (YOLO) | Stage 3A |
| Confidence thresholding to reduce false positives | Stage 3A |
| FPS optimization on CPU | Stage 3A |
| Deploying ML model as a web application | Stage 3B |
| Model caching for production performance | Stage 3B |
| Softmax probability output with confidence display | Stage 3B |

---

## Why each stage exists

**Stage 1 failed on purpose.** Understanding why a basic CNN fails on small data is more valuable than copying a model that works. The failure led directly to the transfer learning solution.

**Stage 2 solves accuracy.** 97.14% on 275 images using only 4,098 trainable parameters proves that pretrained features are more powerful than raw data volume for small-scale tasks.

**Stage 3A solves real-time.** Classification tells you *what* is in an image. Detection tells you *where* and handles multiple objects simultaneously — essential for any live camera application.

**Stage 3B solves deployment.** A model that only runs in a Jupyter notebook has no real-world value. The Streamlit app makes the classifier usable by anyone without a single line of code.

---

## Author

**Sahil Suryawanshi**
Computer Vision Engineer (Fresher) | Pune, Maharashtra
[LinkedIn](http://www.linkedin.com/in/sahil585) · [GitHub](https://github.com/Sahil05-08)
