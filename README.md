# 🛡️ Network Intrusion Detection System (NIDS) using Random Forest - NSL-KDD

This project uses the NSL-KDD dataset to build a Random Forest–based Intrusion Detection System (IDS) that classifies network traffic as either normal or attack. It includes performance evaluation with detailed metrics and a confusion matrix.

## 📦 Features

- 🔁 Auto-downloads the NSL-KDD dataset
- 🔍 Categorical encoding
- 🤖 Random Forest training
- 📊 Classification report (precision, recall, F1-score)
- 🧩 Confusion matrix with interpretation
- 📝 Auto-saves evaluation results to a text file

## 🧰 Requirements

```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas
scikit-learn
matplotlib
seaborn
gdown
```

## 🚀 How to Run

```bash
python nids.py
```

## 📈 Sample Output

```
📋 Classification Report:
              precision    recall  f1-score   support
    attack       1.00      1.00      1.00     17709
    normal       1.00      1.00      1.00     20083
```

```
🧩 Confusion Matrix:
           Predicted
          Attack  Normal
Actual A  20070     13
Actual N     32  17677
```

## 🧠 Output Meaning

- **TP** (True Positive): Attack correctly identified.
- **TN** (True Negative): Normal correctly identified.
- **FP** (False Positive): Normal falsely flagged as attack.
- **FN** (False Negative): Attack missed.

## 🧑‍💻 Author

> Built with ❤️ 

## 📄 License
This project is open-source and available under the [MIT](https://choosealicense.com/licenses/mit/) License.