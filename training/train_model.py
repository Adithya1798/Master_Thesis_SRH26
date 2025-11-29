import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Allow imports from workspace root (utils package)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("...Starting Train Model Script...")

PROCESSED_PATH = os.path.join("data", "processed", "processed_data.csv")
MODEL_PATH = os.path.join("models", "sgd_model.pkl")
CONF_PATH = os.path.join("models", "sgd_confusion_matrix.png")
CHUNK_SIZE = 50000

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)

first_chunk = True
last_X, last_y = None, None

logger.info("...Starting chunk-wise training...")

for chunk in pd.read_csv(PROCESSED_PATH, chunksize=CHUNK_SIZE):
    # Coerce all features to numeric, replace non-convertible values with 0, then cast to float32
    X = chunk.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype("float32")
    # Normalize label strings and map to integers; drop rows with unknown labels
    y = chunk['label'].astype(str).str.strip().map({'normal': 0, 'attack': 1})
    mask = y.notna()
    if not mask.all():
        logger.warning(f"Dropping { (~mask).sum() } rows with unknown labels in this chunk")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype("int8").reset_index(drop=True)

    if first_chunk:
        clf.partial_fit(X, y, classes=np.array([0,1], dtype='int8'))
        first_chunk = False
    else:
        clf.partial_fit(X, y)

    last_X, last_y = X, y

logger.info("...Chunked training complete...")
logger.info("...Evaluating on last batch...")

y_pred = clf.predict(last_X)
acc = accuracy_score(last_y, y_pred)
prec = precision_score(last_y, y_pred, zero_division=0)
rec = recall_score(last_y, y_pred, zero_division=0)
f1 = f1_score(last_y, y_pred, zero_division=0)

logger.info(f"Accuracy: {acc:.4f}")
logger.info(f"Precision: {prec:.4f}")
logger.info(f"Recall: {rec:.4f}")
logger.info(f"F1 Score: {f1:.4f}")

logger.info("\n" + classification_report(last_y, y_pred))

cm = confusion_matrix(last_y, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Attack'],
            yticklabels=['Normal','Attack'])
plt.title("Confusion Matrix (Last Chunk)")
plt.tight_layout()
plt.savefig(CONF_PATH)
plt.close()

logger.info(f"...Confusion matrix saved to {CONF_PATH}...")

joblib.dump(clf, MODEL_PATH)
logger.info(f"...Model saved to {MODEL_PATH}...")
logger.info("...Train Model Script Completed...")
