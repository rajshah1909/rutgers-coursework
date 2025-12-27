# HW3 – Machine Learning  
**Author:** Raj Shah  
**Course:** Machine Learning  
**Assignment:** Homework 3  

---

## 📁 Directory Structure

```
HW3_data/
│   README.md
│   npzfileans.py
│
├── hw3ans2/          # ANSWER 2 – SVM + SMO
├── hw3ans3/          # ANSWER 3 – 1D & 2D GDA + MAP
├── hw3ans4/          # ANSWER 4 – Logistic Regression + PCA + Spam Filter
├── P2_data/          # Iris dataset for Q2
├── P3_data/          # Provided train/test data for Q3
└── P4_files/         # Spam dataset and preprocessing for Q4
```

---

# 2️⃣ Question 2 — Soft-Margin SVM with SMO

### ✔ Implemented:
- Soft-margin SVM dual formulation using **Sequential Minimal Optimization (SMO)**
- Linear kernel:  
  \[
  \kappa(x_n,x_m) = x_n^\top x_m
  \]
- Derived clipping bounds (Equations 13–14)
- Implemented simplified SMO (`train2_2.py`)
- Plotted decision boundary & margins for **C = 1, 10, 100**

### 📄 Files (hw3ans2):
- `train2_2.py` — SMO implementation  
- `svm_output.txt` — training results  
- `svm_plot_C1.png`, `svm_plot_C10.png`, `svm_plot_C100.png` — boundary + margins  
- `q2tex.tex`, `q2tex.pdf` — full LaTeX write-up

### 📊 Results Summary:

| C   | Support Vectors | Margin  |
|-----|------------------|---------|
| 1   | 11               | 0.8438  |
| 10  | 8                | 0.4452  |
| 100 | 8                | 0.6636  |

**All classifiers reach ~99–100% accuracy.  
Decision boundary changes slightly as C varies, but test accuracy remains similar.**

---

# 3️⃣ Question 3 — Naive Bayes, MAP, GDA

### ✔ Implemented:
- 1D Gaussian estimation (`train3_1.py`)
- Decision rule classifier (`test3_2.py`)
- MAP classifier (`test3_3.py`)
- 2D GDA training (`train3_4.py`)
- 2D GDA testing (`test3_5.py`)
- True-density classifier (`test3_6.py`)

### 📄 Output Log Files (auto-saved):

```
test3_2.txt
test3_3.txt
test3_5.txt
test3_6.txt
train3_1.txt
train3_4.txt
```

### 📊 Accuracy Summary:
- Decision Rule Accuracy: **61.50%**
- MAP Accuracy: **90.00%**
- 2D GDA Accuracy: **84.00%**
- True Density Accuracy: **85.00%**

---

# 4️⃣ Question 4 — Logistic Regression, PCA, Spam Filter

### ✔ Implemented:
- Email preprocessing (`preprocessing.py`)
- Vocabulary extraction & vectorization (`data4_2.py`)
- Logistic regression using gradient descent (`train4_3.py`)
- PCA + prediction (`test4_5.py`)

### 📄 Output Log Files:

```
data4_2.txt
train4_3.txt
test4_5.txt
```

### 📊 Summary:
- Logistic Regression Training Accuracy: **~49%**
- PCA Spam Prediction: Produces correct prediction output  
- Model outputs saved to:  
  - `model4_3.npz`
  - `train4_2.npz`
  - `test4_2.npz`

---

# 📝 Notes
- All Python outputs were generated using:
  ```
  python file.py > file.txt
  ```
- All `.npz` files, plots, PDFs, and logs are included.

---

# ✅ Final Submission Complete

This package includes:
- Full source code  
- Execution logs (`.txt`)  
- Plots and PDFs  
- Preprocessing and model files  
- This README.md summarizing everything clearly