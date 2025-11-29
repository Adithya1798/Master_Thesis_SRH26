import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("=== Started Detect Anomaly Script ===")

MODEL_PATH = os.path.join("models", "sgd_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODERS_PATH = os.path.join("models", "label_encoders.pkl")
PROCESSED_PATH = os.path.join("data", "processed", "processed_data.csv")

OUTPUT_DIR = "outputs"
CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
METRICS_FILE = os.path.join(OUTPUT_DIR, "detection_metrics.csv")

os.makedirs(CM_DIR, exist_ok=True)

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

label_map = {'normal': 0, 'attack': 1}
CHUNK_SIZE = 50000

metrics = []
chunk_num = 0

for chunk in pd.read_csv(PROCESSED_PATH, chunksize=CHUNK_SIZE):
    chunk_num += 1
    logger.info(f"Processing Chunk #{chunk_num}...")

    # Coerce features to numeric (handle stray headers or bad rows), fill NaNs
    X_num = chunk.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

    # Normalize and map labels, drop rows with unknown labels
    y_true = chunk['label'].astype(str).str.strip().map(label_map)
    mask = y_true.notna()
    if not mask.all():
        logger.warning(f"Dropping { (~mask).sum() } rows with unknown labels in chunk {chunk_num}")
    X = X_num.loc[mask].reset_index(drop=True)
    y_true = y_true.loc[mask].astype('int8').reset_index(drop=True)

    # Already scaled in preprocess
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"Chunk {chunk_num} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    metrics.append({
        "Chunk": chunk_num,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    })

    # Ensure confusion matrix has shape even if a label is missing in this chunk
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"])
    plt.title(f"Confusion Matrix - Chunk {chunk_num}")
    plt.tight_layout()
    cm_path = os.path.join(CM_DIR, f"chunk_{chunk_num}.png")
    plt.savefig(cm_path)
    plt.close()

    logger.info(f"...Saved confusion matrix to {cm_path}...")

# Save metrics CSV
pd.DataFrame(metrics).to_csv(METRICS_FILE, index=False)
logger.info(f"...Saved detection metrics to {METRICS_FILE}...")
logger.info("=== Detection completed on all data chunks ===")
