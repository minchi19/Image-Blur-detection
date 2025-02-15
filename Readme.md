# Blur Detection Report

## Scope
Given an image, detect whether it is **blurred** or **sharp** using multiple approaches. 

The objective is to analyze different blur detection techniques, compare their performance, and determine the most effective method.

## Types of Blur
### 1. Defocus Blur
- Caused by incorrect focus or depth of field issues.
- Example: Out-of-focus images, Gaussian blur, or low-contrast images.

### 2. Motion Blur
- Occurs due to the movement of the camera or object during exposure.
- Example: Streaking effects in fast-moving scenes.

## Dataset & Preprocessing
- **Dataset Location:** `/content/drive/My Drive/blur_data`
- **Categories:** `Blurred` and `Sharp`
- Total blur images: 1770
- Total sharp images: 2316
- **Preprocessing (for CNN):**
  - Images resized to `(224, 224)`.
  - Normalization (`pixel values scaled to [0,1]`).
  - Data augmentation applied (if applicable).

## Experiments & Analysis

### **Experiment 1: Laplacian Variance (Baseline Model)**
#### **Approach:**
- Compute the **variance of the Laplacian** of the image.
- If the variance is below a set threshold, classify it as blurred.

#### **Results:**
- **Threshold:** 100
- **Accuracy:** 0.74
- **Precision:** 0.65
- **Recall:** 0.85
- **F1-score:** 0.74
- **Confusion Matrix:**
  ```
  [[1523  793]
   [ 270 1500]]
  ```
#### **Inference:**
- Works well for **defocus blur** but struggles with **motion blur**.
- Threshold tuning is required for better performance.

---

### **Experiment 2: Optimized Threshold (Best F1-Score)**
#### **Approach:**
- Optimized the Laplacian variance threshold for better F1-score.

#### **Results:**
- **Threshold:** 75
- **Accuracy:** 0.76
- **Precision:** 0.70
- **Recall:** 0.79
- **F1-score:** 0.74
- **Confusion Matrix:**
  ```
  [[1713  603]
   [ 377 1393]]
  ```
#### **Inference:**
- Slight improvement in recall and precision.
- Still struggles with motion blur cases.

---

### **Experiment 3: Gaussian Mixture Model (GMM) - Discarded**
#### **Approach:**
- Applied GMM to classify sharp vs. blurred images based on Laplacian variance distribution.

#### **Results:**
- **Threshold:** 396.45
- **Accuracy:** 0.53
- **Precision:** 0.48
- **Recall:** 0.98
- **F1-score:** 0.65
- **Confusion Matrix:**
  ```
  [[ 449 1867]
   [  36 1734]]
  ```
#### **Inference:**
- High false positive rate.
- Threshold too high, making the model unreliable.

---

### **Experiment 4: Fourier Transform Method**
#### **Planned Approach:**
- Frequency domain analysis to detect blur based on edge loss.

---

### **Experiment 5.1: CNN from Scratch**
#### **Approach:**
- Built a **custom CNN** model trained on the dataset.
- 4-layer CNN with:

-- Conv2D + ReLU (Feature extraction)
-- MaxPooling (Downsampling)
-- Flatten + Dense (Final classification)
--

#### **Results:**
```
              precision    recall  f1-score   support

       Sharp       0.81      0.88      0.84       327
     Blurred       0.84      0.76      0.80       286

    accuracy                           0.82       613
   macro avg       0.83      0.82      0.82       613
weighted avg       0.82      0.82      0.82       613
```
#### **Inference:**
- Outperforms traditional methods.
- Requires a significant amount of training data.

---

### **Experiment 5.2: Transfer Learning (Ongoing)**
#### **Approach:**
- Fine-tuned a **pretrained MobileNetV2 model** for blur classification.
- 

## Summary of Results
| Experiment | Accuracy | Precision | Recall | F1-score |
|------------|----------|-----------|--------|----------|
| Laplacian Variance (Baseline) | 0.74 | 0.65 | 0.85 | 0.74 |
| Optimized Threshold | 0.76 | 0.70 | 0.79 | 0.74 |
| GMM (Discarded) | 0.53 | 0.48 | 0.98 | 0.65 |
| CNN from Scratch 4 Layers| 0.82 | 0.83 | 0.82 | 0.82 |
| Transfer Learning Using MobileNetV2| *Training in Progress* | *TBD* | *TBD* | *TBD* |

## Challenges Faced
- **High runtime memory usage:** Initially caused crashes in Google Colab.
- **Class imbalance:** Some methods had a high false positive rate.
- **Choosing the right threshold:** Required optimization.
- **Motion vs. Defocus Blur:** Some methods failed to differentiate between these.

## Next Steps
- Explore real-time blur detection applications.
- Investigate object-aware blur detection for more granular insights.

## References
- https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
- https://github.com/Utkarsh-Deshmukh/Spatially-Varying-Blur-Detection-python
---
**Conclusion:** This study explores multiple methods for blur detection, highlighting their strengths and weaknesses. 
The CNN and Transfer Learning approaches are expected to outperform traditional threshold-based techniques.

