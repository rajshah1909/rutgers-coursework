# 📊 Introduction to Data Science – CS 439 (01:198:439), Spring 2026

Jupyter notebooks for the CS 439 labs and homeworks, with the datasets needed to re-run them.

## 🧰 Tools
Python 3, NumPy, Pandas, Matplotlib/Seaborn, SciPy (KDE)

## 📂 Structure
| Folder | Topic | Notebook | Data |
|---|---|---|---|
| `HW01` | Working with NumPy and Pandas | `Lab01_Numpy_and_Pandas.ipynb` | `StudentsPerformance.csv` |
| `HW02` | Naive Bayes SMS spam classifier | `Naive_Bayes_Spam.ipynb` | `SMSSpamCollection.txt` |
| `HW03` | Data visualization and kernel density estimation | `Data_Visualization_and_KDE.ipynb` | `student_performance.csv`, `daily_market_index.csv`, `household_pets.csv` |
| `HW04` | Logistic regression from scratch for spam detection | `Logistic_Regression_Spam.ipynb` | `SMSSpamCollection.txt` |
| `HW05`–`HW10` | Not yet released | — | — |

## ▶️ Run
```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
jupyter notebook
```
Open a notebook from its own `HWnn/` folder so the relative data paths resolve.
