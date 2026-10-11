# 🏦 Loan Approval Prediction

A machine learning project that predicts whether a loan will be **Approved** or **Not Approved**. The final model is shown in an interactive web app built with Gradio.

## 📊 Dataset
 **Source:** [Loan Approval Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
Loan approval dataset from Kaggle (about 45,000 records).
Target column: `Loan Status` (1 = Approved, 0 = Not Approved).

## ⚙️ What I Did
1. Cleaned the data and filled missing values
2. Converted text columns into numbers
3. Split the data (80% train, 20% test)
4. Scaled the features
5. Trained and compared 5 models
6. Built a Gradio web app

## 🤖 Results

| Model | Accuracy |
|---|---|
| **Random Forest** | **91.53%** |
| SVC | 90.13% |
| Logistic Regression | 89.09% |
| KNN | 88.99% |
| Decision Tree | 87.51% |

Random Forest gave the best result.

## 🛠️ Tools Used
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Gradio, Google Colab , model.pkl

## Loan_approval (output) link
https://18d9b7a5bbcce9782f.gradio.live/

## 👩‍💻 Author
**Drashti Gohel**
[GitHub](https://github.com/drashtitech09)
