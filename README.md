# Sisuni Virtual Try-On System 👕
## How Actual Virtual Try-On Works

In real virtual try-on research, the workflow typically follows these steps:

### 1. Pose Estimation / Landmark Detection  
*(e.g., MediaPipe, OpenPose)*  
- Extract keypoints (shoulders, hips, torso, arms, etc.).  
- Define a bounding box or region where the shirt should be placed.  

---

### 2. Cloth Warping / Geometric Matching  
*(GMM, TPS warping)*  
- Warp the shirt to match the person’s pose & body shape.  
- Ensures sleeves align with arms, neckline with shoulders.  

---

### 3. Masking & Parsing  
*(Human Parsing models like SCHP, CIHP)*  
- Segment the person into regions (torso, arms, legs, background).  
- Create a mask where the shirt should go.  

---

### 4. Composition Network  
*(UNet / ResNet Generator)*  
- Inputs: **(person image, warped shirt, mask)**  
- Outputs: a realistic try-on image (blends colors, shadows, textures).  

---

### 5. Refinement *(Optional)*  
- Use GAN loss, perceptual loss, or refinement modules to improve realism.  

*This pipeline demonstrates how actual virtual try-on models generate realistic clothing overlays on a person.*

## My Implementation

## 👕 Overlay using MediaPipe

This project implements a **virtual try-on system** where a t-shirt is overlaid onto a person’s image using **MediaPipe Pose Landmarker** and **OpenCV**.  

The system detects key body landmarks (shoulders and hips), calculates the appropriate size and position of the t-shirt, and blends it naturally on the person’s upper body using segmentation masks and alpha transparency.

---

## 🖥️ Simple CNN Model with One-Sample Training

This section demonstrates a **simple convolutional neural network (CNN)** trained on a single sample for virtual try-on. The model takes a person image and a t-shirt image as input and predicts the resulting try-on image.

**Key points:**
- **Input:** Concatenation of person image and t-shirt image (6 channels).  
- **Model:** Small UNet-like CNN with encoder, middle, and decoder blocks.  
- **Training:** Single sample training using L1 loss for 1500 epochs.  
- **Output:** Predicted try-on image showing the t-shirt overlaid on the person.  

This simple setup serves as a **proof of concept** for end-to-end try-on prediction using deep learning.

---

## 🖥️ Simple CNN Model with Three-Sample Training

This section extends the previous single-sample training to a slightly larger dataset with **three samples**. The goal is to train a Simple UNet-like CNN to predict virtual try-on results from multiple inputs.

**Key points:**
- **Input:** Concatenation of person image and t-shirt image (6 channels).  
- **Dataset:** Three image samples, each containing a person image, a t-shirt image, and the corresponding target try-on image.  
- **Model:** Same UNet-like CNN with encoder, middle, and decoder blocks.  
- **Training:** L1 loss with Adam optimizer for 1500 epochs, using a DataLoader for batching.  
- **Output:** Predicted try-on image for any input sample after training, showing improved generalization compared to single-sample training.

This demonstrates training on a small dataset and highlights how the model begins to **learn patterns across multiple samples** for more realistic virtual try-on results.

---

## ⚙️ How to Run

1. **Upload to Google Colab:**  
   - Upload the Jupyter Notebook (`.ipynb`) to Colab.  
   - Upload all images, t-shirts, target outputs, and the `pose_landmarker.task` model in a zip file and extract them in Colab.

2. **Install dependencies:**  
   - All required module installation commands are included in the notebook:  
     ```bash
     pip install opencv-python mediapipe torch torchvision numpy matplotlib pillow
     ```  
   - Just run the cells in Colab.

3. **Run the notebook:**  
   - Click **Run All** in Colab.  
   - The virtual try-on system will overlay t-shirts and train the CNN models.  
   - Output images will be displayed inline and saved in the notebook folder.

---

## 📷 Output

- `tryon.jpg` → Person image with the t-shirt virtually applied.  
- Predicted try-on images from **single-sample CNN** and **three-sample CNN** training.

---
# 👕 Virtual Try-On Prototype (3 Samples)

This project is a **prototype for a virtual try-on system** that overlays shirts onto images of people using **MediaPipe for pose detection** and a **UNet-based refinement network**.  
It demonstrates the pipeline with **3 sample images**.

---

## 🚀 Features
- **Pose Detection** (using MediaPipe Pose landmarks).
- **Cloth Warping** based on body keypoints (shoulders & hips).
- **Overlay & Masking** to align shirt images onto the person.
- **Custom Dataset Loader** for training.
- **Simple UNet Model** to refine shirt-person overlay.
- **Training Loop** with L1 loss optimization.
 in the next cell you can find the output images 
---
