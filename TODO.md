# 🛠️ TODO — Playground AI Pipeline (v0.0)

This is the development roadmap for the Playground project: a modular framework for building, training, evaluating, and managing AI workflows across multiple frameworks (e.g., Hugging Face, Ultralytics, Roboflow).

---

## 🔍 1. Annotation & Data Cleaning

- [ ] Add **auto-annotation** (predict annotations using pretrained models)
- [ ] Implement **auto-cleaning pipeline**:
  - [ ] Use unsupervised clustering to detect duplicates/anomalies
  - [ ] Integrate **CBIR** (content-based image retrieval)
  - [ ] Incorporate **pretrained embeddings** for similarity metrics
- [ ] Integrate **CleanLab** for noise detection and label error correction

---

## ☁️ 2. Cloud Integration

- [ ] Create a unified **CloudConnector** class
- [ ] Integrate **CVAT** for annotation management
- [ ] Add support for:
  - [ ] AWS (S3, Lambda)
  - [ ] GCP (Cloud Storage)
  - [ ] Azure (Blob Storage)
- [ ] Add support for remote model training & deployment

---

## ⚙️ 3. Core Features

### Common Utilities
- [ ] Add centralized **logging utility** (structured, color-coded logs)
- [ ] Add custom **exception classes** with context-aware errors

### Task Orchestration
- [ ] Improve `task.py` runner to support:
  - [ ] Multi-step pipelines
  - [ ] Validation of config structure
  - [ ] Timing and benchmarking per stage

---

## 🧩 4. Dataset Support

- [ ] Support datasets for:
  - [ ] Object detection (YOLO, COCO, VOC)
  - [ ] Classification (ImageNet-like)
  - [ ] Oriented bounding boxes (OBB)
  - [ ] Human pose estimation (e.g., COCO keypoints)
  - [ ] Semantic segmentation
- [ ] Add unit tests for dataset adapters

---

## ⬇️ 5. Download System

- [ ] Build a **download factory**:
  - [ ] Support downloads from Hugging Face, Roboflow, CVAT, URLs, or cloud buckets
  - [ ] Add metadata validation and format normalization

---

## 📦 6. Framework Integration

- [x] Integrate Ultralytics (YOLOv8) for train/eval/export
- [x] Integrate Hugging Face (transformers, datasets)
- [ ] Add Roboflow connector
- [ ] Define abstract interface for adding more frameworks

---

## 📥 7. Input Pipeline

- [x] Support folder-based input (for detection)
- [ ] Add support for CSV-based input (for all tasks)
- [ ] Add input pipelines for:
  - [ ] OBB
  - [ ] Classification
  - [ ] Segmentation
  - [ ] Pose
  - [ ] Text input (plain files or CSV)
- [ ] Define a **standard format for text data** (e.g., input/label JSON)
- [ ] Add input validation and error reporting

---

## ⚙️ 8. Processing Pipeline

- [x] Add basic processor (rename, resize)
- [x] Support linear job composition
- [ ] Add job types:
  - [ ] Cropping
  - [ ] Masking
  - [ ] Image augmentation
  - [ ] Tiling for large images
- [ ] Add parallel processing or multiprocessing option
- [ ] Add metadata logging per processing step

---

## 🧠 9. Models

- [x] Support model training, prediction, evaluation
- [x] Add benchmarking with timing and metrics
- [x] Add export functionality (ONNX, TorchScript, etc.)
- [ ] Store benchmark results in structured format (JSON or CSV)
- [ ] Track training parameters, configs, and logs
- [ ] Detect available devices (GPU, MPS, CPU) dynamically
- [ ] Add device routing in config or CLI

---

## 📊 10. Reporting

- [ ] Generate automated evaluation reports (HTML or PDF)
- [ ] Include:
  - [ ] Confusion matrices
  - [ ] Precision-recall curves
  - [ ] Class-wise metrics
  - [ ] Visual examples (input + prediction overlay)

---

## 📤 11. Storage

- [x] Implement basic save/load storage interface
- [x] Add support for versioned file storage
- [ ] Add hash-based integrity check
- [ ] Add support for saving to remote locations (cloud, NFS, etc.)

---

## 🚀 12. CLI & Interface

- [ ] Add a `cli.py` using [Typer](https://typer.tiangolo.com/) or `argparse`:
  - [ ] `train`, `predict`, `evaluate`, `process`, `export`, etc.
- [ ] Example:
  ```bash
  python cli.py train --config configs/train.yaml
  python cli.py annotate --input data/images --model yolov8n
