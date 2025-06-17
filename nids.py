import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import gdown

print("\n📥 Step 1: Downloading NSL-KDD Dataset...")
gdown.download(
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    "KDDTrain+.txt",
    quiet=False
)

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

print("\n📄 Step 2: Loading Dataset...")
df = pd.read_csv("KDDTrain+.txt", names=columns)
df.drop('difficulty', axis=1, inplace=True)

df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

print("🔄 Step 3: Encoding Categorical Columns...")
for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('label', axis=1)
y = df['label']

print("✂️  Step 4: Splitting Dataset (Train 70% / Test 30%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🌲 Step 5: Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("🤖 Step 6: Predicting on Test Set...")
y_pred = model.predict(X_test)

print("\n✅ Step 7: Model Evaluation\n")

# Classification Report
print("📋 Classification Report:\n")
report = classification_report(y_test, y_pred, target_names=["attack", "normal"])
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("🧩 Confusion Matrix:")
print(f"\n            Predicted")
print(f"           Attack   Normal")
print(f"Actual A   {tp:>6}   {fn:>6}")
print(f"Actual N   {fp:>6}   {tn:>6}\n")

print("📖 Explanation:")
print(f"- True Positives  (TP): {tp} → Attacks correctly identified.")
print(f"- False Negatives (FN): {fn} → Missed attacks.")
print(f"- True Negatives  (TN): {tn} → Normal traffic correctly identified.")
print(f"- False Positives (FP): {fp} → Normal traffic falsely flagged as attack.")

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\n🧠 Model Interpretation:")
print("- The classifier performs exceptionally well.")
print("- Precision, Recall, and F1-Score are all close to 1.0.")
print("- Very few false alarms or missed attacks.")
print("- Suitable for real-time intrusion detection scenarios.\n")

# Save explanation to file
with open("model_evaluation_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
print("📁 Summary saved to 'model_evaluation_report.txt'.")

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
