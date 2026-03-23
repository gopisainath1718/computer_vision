# Computer Vision Projects Repository

This repository contains a collection of computer vision projects implemented in **C++** (using OpenCV) and **Python** (using PyTorch). These projects cover a wide range of computer vision tasks, from camera calibration and augmented reality to content-based image retrieval, real-time filtering, and deep learning for digit recognition.

## Projects Overview

1. **Calibration-Augmented Reality (`Calibration-Augmented_Reality`)**
   - Implements camera calibration using a checkerboard pattern and projects 3D virtual objects onto the 2D image plane to create Augmented Reality (AR) effects.
2. **Content-Based Image Retrieval (`Content_based_image_retrieval`)**
   - An image search engine based on content features (like color histograms, spatial features, texture, etc.). Custom distance metrics are computed and stored in CSV files to find similar images from a dataset.
3. **Real-Time 2D Object Recognition (`Real-time_2-D_object_recognition`)**
   - Performs real-time object detection and recognition from a live camera feed. Uses thresholding, connected components, and feature extraction to identify and classify objects.
4. **Real-Time Filtering (`Real-time_filtering`)**
   - Applies various real-time video filters (e.g., blurring, edge detection, quantization) using customized filter kernels and OpenCV on live web-camera streams.
5. **Recognition Using Deep Networks (`Recognition_using_deep_networks`)**
   - Evaluates digit recognition on the MNIST dataset and custom greek letters. Contains scripts (`Task-*.py`) to train, analyze, and run a ResNet-18 model or simpler CNNs using PyTorch.

---

## Prerequisites

### C++ Projects Requirements
- **C++ Compiler**: `g++` (Linux) or MSVC (Windows Visual Studio).
- **OpenCV 4.x**: Must be installed and configured on your system.

### Python projects Requirements
- **Python 3.8+**
- **Libraries**:
  - `torch` (PyTorch)
  - `torchvision` (for MNIST and simple transformations)
  - `numpy`
  - `matplotlib`
  - `opencv-python` (cv2)

To install the Python dependencies:
```bash
pip install torch torchvision numpy matplotlib opencv-python
```

---

## How to Clone and Setup

First, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/computer_vision.git
cd computer_vision
```

---

## Running Instructions

### Running C++ OpenCV Projects (Linux/macOS)
Navigate into any C++ project directory (e.g., `Real-time_filtering`):
```bash
cd Real-time_filtering
```

Compile the source code using `g++`. For example, compiling the main function along with utility files:
```bash
# Example compilation command. Ensure you link all necessary local .cpp files
g++ filter.cpp vidDisplay.cpp imgDisplay.cpp -o app `pkg-config --cflags --libs opencv4`
```

Then run the compiled executable:
```bash
./app
```
*(Note for Windows users: A Visual Studio project `.vcxproj` might be provided in some directories (like `Real-time_2-D_object_recognition`). You can open it and compile via Visual Studio.)*

### Running Python Deep Learning Projects
Navigate into the deep learning project folder:
```bash
cd Recognition_using_deep_networks
```

Run the individual tasks:
```bash
python Task-1.py
```
*Depending on the script, the MNIST dataset will automatically download using `torchvision.datasets.MNIST`. Ensure you have an active internet connection on the first run. The trained model weights (e.g., `resnet18_mnist.sav`) might also be required for live recognition tasks.*

---

## Datasets and Assets
- **Images/Videos For Filtering & AR**: The sample images (e.g., `image_1.jpg`) and short videos (e.g., `1.mp4`) are provided in the respective directories.
- **Image Retrieval Data**: The baseline image dataset (e.g. `olympus`) must be placed inside the `Content_based_image_retrieval` directory if you intend to recalculate the feature vectors and rewrite the CSVs.
- **Deep Networks Models/Data**: Included test datasets (`greek_test`, `greek_train`, `numbers`) are in the `data` subfolder or root folder of the deep networks project. Pre-trained weights like `resnet18_mnist.sav` and `model.sav` are present for immediate execution of inference tasks.
