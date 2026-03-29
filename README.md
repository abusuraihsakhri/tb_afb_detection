# 🔬 TB-AFB Clinical Intelligence Engine
### Developed by **Dr. Abu Suraih Sakhri**

[![Clinical Grade](https://img.shields.io/badge/Status-Clinical--Ready-success.svg)](https://github.com/abusuraihsakhri/tb_afb_detection)
[![Cyber Security](https://img.shields.io/badge/Audit-PASSED-blue.svg)](SECURITY.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

A state-of-the-art, **GPU-accelerated Whole Slide Image (WSI) detection platform** for Mycobacteria Tuberculosis (AFB) screening. This system is designed for high-throughput digital pathology, providing automated diagnostic assistance and active learning capabilities.

---

## 👨‍💻 Author Info
This system was developed and architected by **Dr. Abu Suraih Sakhri** with a focus on bridging the gap between clinical pathology expertise and advanced neural orchestration.

---

## 🚀 Key Clinical Capabilities
*   **Universal Hardware Acceleration**: 
    *   **NVIDIA**: Full CUDA support for Tensor Cores.
    *   **Apple Silicon**: Native **Metal/MPS** support for Mac **M-Series (M1, M2, M3, M4 and beyond)**.
    *   **Hybrid Orchestration**: Automatically scales from personal workstations (8GB RAM) to high-capacity clinical servers (128GB+ RAM).
*   **Multi-Platform Ingestion**: Full support for Windows 10/11, macOS (Ventura+), and Generic Linux (Ubuntu/Debian/Fedora).
*   **High-Speed WSI Tiling**: Multi-threaded slicing of massive pyramidal slides (.svs, .ndpi, .tiff).
*   **Dynamic Weight Orchestration**: Instant adoption of updated neural training results without service interruption.
*   **Active Learning Engine**: Integrated clinical annotation tool for expert-driven algorithm optimization.

---

## 🧠 How it Works: Component Deep-Dive

### 1. WSI Slicing Engine (Inpainter/Extractor)
Medical Whole Slide Images (WSIs) often exceed 100,000 pixels in dimension, making them too large for direct neural processing. 
*   **The Logic**: The system utilizes **OpenSlide** and **libVIPS** to slice the massive binary file into uniform 512x512 "tiles". 
*   **Clinical Intelligence**: To save time and compute, the system performs a **Background Masking** check, automatically skipping empty glass/white space to process only those tiles containing relevant tissue.

### 2. Neural Orchestration & Hot-Swapping
The core of the detection is a YOLO-based deep learning architecture.
*   **Detection**: The model analyzes each tissue tile for the specific morphology of M. Tuberculosis (rod-like structure, ZN-staining characteristics).
*   **Device Mapping**: It automatically detects your hardware. On specialized workstations, it uses **CUDA**. On modern Macs, it uses **Metal (MPS)**.
*   **Hot-Swapping**: When a pathologist finishes a new training run, the API detects the updated weights (`best.pt`) and "Hot-Mounts" them instantly.

### 3. Active Learning & Expert Ingestion
The system allows the algorithm to grow smarter with every slide review.
*   **Ground Truth**: Pathologists use the **Annotation Tool** to manually box valid AFB rods. 
*   **Data Hygiene**: These boxes are converted into standardized YOLO coordinates and ingested with an automated **20% Validation Split**.
*   **Retraining**: A single click initiates a background training thread that fine-tunes the existing model.

---

## 🛠️ Infrastructure & Hardware Scaling Guide

The TB-AFB Engine is designed to scale dynamically based on your available hardware.

| Analysis Target | Recommended RAM | Recommended Compute |
| :--- | :--- | :--- |
| **Microscope Patches (JPG/PNG)** | 8GB - 16GB | 4-Core CPU / Any GPU |
| **Standard WSI (1GB - 5GB .svs)** | 32GB - 64GB | 8-Core CPU / 8GB VRAM GPU |
| **High-Res Pyramidal (10GB+ .ndpi)** | 128GB+ | 16-Core+ CPU / 16GB+ VRAM GPU |

### Minimum Requirements
*   **OS**: Windows 10/11, macOS Ventura+, or Ubuntu 22.04+
*   **CPU**: Any modern 64-bit multi-core processor.
*   **RAM**: 8GB Minimum (16GB+ highly recommended for active learning).
*   **GPU**: Optional (CUDA 12+ or Apple Silicon M-Series for acceleration).
*   **Drivers**: OpenSlide 3.4.1+ (Mandatory for WSI processing).

### OS-Specific Setup (Install Medical Drivers)
*   **macOS**: `brew install openslide`
*   **Ubuntu/Debian**: `sudo apt install libopenslide0-dev`
*   **Windows**: Download binaries from [OpenSlide.org](https://openslide.org/download/)

---

## 🏃 Usage & Clinical Inference

The system is now fully functional with an integrated secure inference pipeline.

### Running Diagnostic Screening
To analyze a Whole Slide Image (WSI) and generate a WHO-compliant report, use the localized `04_inference.py` script:

```bash
# Windows / macOS / Linux
python 02_CODE/scripts/04_inference.py --model yolov8n.pt --wsi raw_wsi/your_slide.svs --conf 0.25
```

### Interpreting the Diagnostic Report
The engine outputs a standardized clinical quantitation:
*   **Result**: Negative, Scanty, 1+, 2+, or 3+.
*   **AFB Count**: Absolute number of identified bacilli.
*   **Coverage**: Total tissue fields (HPF equivalent) analyzed.

---

## 🔬 Component Architecture (Verified)

1.  **Sliding Window Orchestrator**: Multi-threaded extraction of WSI regions with strict memory jailing.
2.  **Neural Detector**: YOLOv8-based morphology recognition optimized for ZN-staining.
3.  **Secure Postprocessor**: Vectorized Non-Maximum Suppression (NMS) and aspect-ratio validation to filter staining artifacts.
4.  **WHO Grader**: Automated calculation of IUATLD/WHO smear grading scales.

---

## 🛡️ Cyber Security & Data Privacy
Designed with a **Security-First** mindset:
1.  **Airtight Datalake Isolation**: All file operations are strictly jailed to the `01_DATA` directory via `secure_file_resolution`.
2.  **Verified Checkpoint Guard**: Implements `weights_only=True` loading blocks to prevent arbitrary code execution (RCE).
3.  **Resource Guard**: Implements `try...finally` resource management to prevent File Descriptor exhaustion and DoS attacks.
4.  **Privacy-First Datalake**: Built-in protection ensures no patient data is exposed to version control.

## ⚖️ License
Distributed under the **Apache License 2.0**. See `LICENSE` for more information.

---
*For clinical inquiries or technical support, please refer to the documentation or contact the developer.*
