import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── 1. Load data ──────────────────────────────────────────────────────────────
abstracts = pd.read_csv("data/abstracts.csv")[["pmid", "abstract"]]
features = pd.read_csv("data/extracted_features.csv")
df = features.merge(abstracts, on="pmid", how="inner")

# Drop rows with missing abstract or label
df = df.dropna(subset=["abstract", "study_type"])

print(f"Training on {len(df)} abstracts")
print(f"\nClass distribution:\n{df['study_type'].value_counts()}")

# ── 2. Vectorize text with TF-IDF ─────────────────────────────────────────────
# Convert raw abstract text into a matrix of TF-IDF features
# max_features=500 keeps the 500 most informative words
vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(df["abstract"])
y = df["study_type"]

# ── 3. Train/test split ───────────────────────────────────────────────────────
# 80% train, 20% test, stratified so class proportions are preserved
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 4. Train logistic regression classifier ───────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 6. Confusion matrix ───────────────────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Study Type Classifier")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved.")

# ── 7. Top predictive words per class ────────────────────────────────────────
print("\nTop 10 words predictive of each study type:")
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(model.classes_):
    top_indices = model.coef_[i].argsort()[-10:][::-1]
    top_words = [feature_names[j] for j in top_indices]
    print(f"  {class_label:<20} {', '.join(top_words)}")

# ── 8. Save model and vectorizer ──────────────────────────────────────────────
with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("data/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("\nModel and vectorizer saved.")