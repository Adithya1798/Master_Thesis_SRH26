import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Allow imports from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("...Starting Pre-Process Data Script...")

RAW_PATH = os.path.join("data", "raw", "kddcup.data_10_percent.gz")
PROCESSED_PATH = os.path.join("data", "processed", "processed_data.csv")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODERS_PATH = os.path.join("models", "label_encoders.pkl")

CHUNK_SIZE = 50000

COLUMN_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

categorical_cols = ['protocol_type','service','flag']
numeric_cols = [c for c in COLUMN_NAMES if c not in categorical_cols + ['label']]

# Step 1: Fit label encoders
logger.info("...Step 1: Collect unique categorical values and fit encoders...")

encoders = {col: LabelEncoder() for col in categorical_cols}
unique_vals = {col: set() for col in categorical_cols}

for chunk in pd.read_csv(RAW_PATH, names=COLUMN_NAMES, chunksize=CHUNK_SIZE, compression='gzip', header=None):
    for col in categorical_cols:
        unique_vals[col].update(chunk[col].astype(str).unique())

for col in categorical_cols:
    encoders[col].fit(list(unique_vals[col]))

logger.info("...Encoders fitted on categorical values...")

# Step 2: Fit scaler manually using running mean/variance
logger.info("...Step 2: Fit StandardScaler on numeric columns...")

count = 0
mean = np.zeros(len(numeric_cols), dtype=np.float64)
M2 = np.zeros(len(numeric_cols), dtype=np.float64)

for chunk in pd.read_csv(RAW_PATH, names=COLUMN_NAMES, chunksize=CHUNK_SIZE, compression='gzip', header=None):
    nums = chunk[numeric_cols].astype(np.float64).fillna(0.0).values
    for row in nums:
        count += 1
        delta = row - mean
        mean += delta / count
        delta2 = row - mean
        M2 += delta * delta2

var = M2 / (count - 1 if count > 1 else 1)
scale = np.sqrt(var)

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = scale
scaler.var_ = var

logger.info("...Scaler parameters computed...")

# Step 3: Transform chunks & save processed CSV
logger.info("...Step 3: Transforming and saving data in chunks...")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

first_write = True
for chunk in pd.read_csv(RAW_PATH, names=COLUMN_NAMES, chunksize=CHUNK_SIZE, compression='gzip', header=None):
    for col in categorical_cols:
        chunk[col] = encoders[col].transform(chunk[col].astype(str))

    nums = chunk[numeric_cols].astype(np.float64).fillna(0.0).values
    nums_scaled = (nums - scaler.mean_) / (scaler.scale_ + 1e-9)
    chunk.loc[:, numeric_cols] = nums_scaled

    chunk['label'] = chunk['label'].apply(lambda x: 'normal' if str(x).strip() == 'normal.' else 'attack')

    chunk.to_csv(PROCESSED_PATH, mode='a', header=first_write, index=False)
    first_write = False

    logger.info(f"...Wrote processed chunk to {PROCESSED_PATH}...")

# Save artifacts
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(encoders, ENCODERS_PATH)

logger.info(f"...Scaler saved to {SCALER_PATH}...")
logger.info(f"...Label encoders saved to {ENCODERS_PATH}...")
logger.info("...Pre-Process Data Script Completed...")
