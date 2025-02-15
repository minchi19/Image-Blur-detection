# Blur Detection 
Code - https://colab.research.google.com/drive/19ZFVE0RuJ4wJ5u14tEQyC2HDk3Nv32Z8?usp=sharing

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
### Evaluation Metrics?
Since blur detection is a binary classification task, the F1 score is preferred over accuracy, since the dataset is imbalanced.

### **Experiment 1: Laplacian Variance (Baseline Model)**
#### **Approach:**
- Compute the **variance of the Laplacian**  (second derivative) of an image.
- If the variance is below a set threshold (variance < 100), classify it as blurred.

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
- The threshold is too high, making the model unreliable.
  **Why this was discarded?**
- Overly high threshold misclassifies most sharp images as blurred.
---

### **Experiment 4: Fourier Transform Method (Frequency Analysis)
**
#### **Planned Approach:**
- Frequency domain analysis to detect blur based on edge loss.

---

### **Experiment 5.1: CNN from Scratch**
#### **Approach:**
- Built a **custom CNN** model trained on the dataset.
### Steps:
1. **Dataset Preparation:**
   - Used ImageDataGenerator for augmentation (rotation, flipping, brightness changes).
   - Train-validation-test split: **2753-590-590**.

2. **Model Architecture:**
   - Convolutional layers with ReLU activation. Conv2D + ReLU (Feature extraction)
   - Max pooling for feature extraction.
   - Fully connected dense layers with dropout.
   - Softmax activation for classification.

3. **Loss Function & Optimizer:**
   - Loss: **Categorical Crossentropy**
   - Optimizer: **Adam (learning rate = 1e-3)**

4. **Evaluation:**
- **Accuracy:** 0.82  
- **F1-score:** 0.82  

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

### **Experiment 5.2: Transfer Learning**
#### **Approach:**
- Fine-tuned a **pretrained MobileNetV2 model** for blur classification.
**Why MobileNetV2?**
- Lightweight model designed for mobile and edge applications.
- Pretrained on ImageNet for feature extraction.

###  Steps:
1. **Load MobileNetV2 without top layers.**
2. **Freeze base model layers to retain pre-trained features.**
3. **Add a custom classifier:**
   - Global Average Pooling layer
   - Fully connected dense layers
   - Softmax output layer
4. **Train model with fine-tuning:**
   - First phase: Train top layers only.
   - Second phase: Unfreeze the last 10 layers for fine-tuning with a lower learning rate.

### Results:
- **Accuracy:** 0.86  
- **F1-score:** 0.86  

 **Key Takeaway:**
- Performs better than CNN from scratch.
- Generalizes well despite fewer training images.

- 

## **Summary Table**
| Experiment | Accuracy | Precision | Recall | F1-score |
|------------|----------|----------|--------|----------|
| Laplacian Variance | 0.74 | 0.65 | 0.85 | 0.74 |
| Optimized Threshold | 0.76 | 0.70 | 0.79 | 0.74 |
| GMM | 0.53 | 0.48 | 0.98 | 0.65 |
| CNN from Scratch | 0.82 | 0.82 | 0.82 | 0.82 |
| MobileNetV2 | 0.86 | 0.86 | 0.86 | 0.86 |

---
## Challenges Faced
- **High runtime memory usage:** Initially caused crashes in Google Colab.
- **Class imbalance:** Some methods had a high false positive rate.
- **Choosing the right threshold:** Required optimization.
- **Motion vs. Defocus Blur:** Some methods failed to differentiate between these.

## **Next Steps & Improvements**
1. **Try Different Models:**
   - Train **ResNet50** or **EfficientNet** to compare results.
   
2. **Increase Training Time:**
   - Train CNN for **more epochs** to improve learning.

3. **Improve Augmentation:**
   - Use stronger augmentation techniques (Gaussian noise, motion blur simulation).

4. **Experiment with New Loss Functions:**
   - Try **Focal Loss** to handle class imbalance.


## References
- https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
- https://github.com/Utkarsh-Deshmukh/Spatially-Varying-Blur-Detection-python
---
**Conclusion:** This study explores multiple methods for blur detection, highlighting their strengths and weaknesses. 
From simple thresholding methods to deep learning, we explored multiple approaches for blur detection. 
While MobileNetV2 performed the best, future improvements can enhance robustness.

---

