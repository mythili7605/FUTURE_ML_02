# 🎫 Support Ticket Classification System

## 📌 Project Overview

The Support Ticket Classification System is a Machine Learning and NLP-based application designed to automatically classify customer support tickets into predefined categories and assign priority levels (High, Medium, Low).

This system helps support teams streamline ticket management, reduce response times, and improve customer satisfaction.

---

## 🎯 Objectives

- Automatically classify support tickets into categories.
- Assign priority levels based on ticket content.
- Perform text preprocessing and feature extraction.
- Evaluate model performance using classification metrics.
- Generate insights for support analytics.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- NLTK / spaCy
- Matplotlib
- Seaborn

---

## 📂 Dataset

The dataset contains customer support tickets with the following fields:

| Column | Description |
|----------|-------------|
| Ticket_ID | Unique ticket identifier |
| Ticket_Text | Customer complaint/request |
| Category | Ticket category |
| Priority | High / Medium / Low |

### Example

| Ticket_Text | Category | Priority |
|-------------|----------|----------|
| Internet is not working since morning | Technical Issue | High |
| Need help updating my profile | Account Support | Medium |
| How can I change my password? | Account Support | Low |

---

## ⚙️ Project Workflow

### 1. Data Collection
- Gather support ticket data.
- Store data in CSV format.

### 2. Text Preprocessing
- Convert text to lowercase.
- Remove punctuation and special characters.
- Remove stopwords.
- Tokenization.
- Lemmatization/Stemming.

### 3. Feature Extraction
- TF-IDF Vectorization.
- Convert text into numerical features.

### 4. Model Training
Algorithms tested:
- Naive Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### 5. Ticket Classification
Predict ticket categories such as:
- Technical Issue
- Billing Issue
- Account Support
- Service Request
- Complaint

### 6. Priority Assignment
Assign:
- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority

### 7. Model Evaluation
Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📊 Sample Output

### Input Ticket
```
My payment was deducted but the subscription was not activated.
```

### Predicted Output
```
Category: Billing Issue
Priority: High
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/mythili7605/FUTURE_ML_02.git
```

### Navigate to Project Folder

```bash
cd FUTURE_ML_02
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

---

## 📈 Results

- Automated support ticket categorization.
- Reduced manual ticket sorting effort.
- Faster identification of critical issues.
- Improved support team productivity.

---

## 📁 Project Structure

```text
Support-Ticket-Classification/
│
├── dataset/
│   └── support_tickets.csv
│
├── notebooks/
│   └── Support_Ticket_Classification.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── requirements.txt
│
├── README.md
│
└── app.py
```

---

## 🔮 Future Enhancements

- Deep Learning models (LSTM, BERT)
- Real-time ticket classification
- Dashboard using Streamlit
- Multi-language support
- Sentiment analysis integration

---

## 👩💻 Author

**Mythili**

Aspiring AI Engineer | Machine Learning Enthusiast

---

## 📜 License

This project is open-source and available under the MIT License.