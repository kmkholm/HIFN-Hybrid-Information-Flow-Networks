# =========================================================
# 0) (Colab) Install dependencies
# =========================================================
!pip -q install kagglehub pytorch-tabnet imbalanced-learn scikit-learn

import os
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, hamming_loss,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)

import matplotlib.pyplot as plt

from pytorch_tabnet.tab_model import TabNetClassifier

# Optional (nicer confusion matrix)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)




# =========================================================
# 1) Download dataset using kagglehub (YOUR requested part)
# =========================================================
import kagglehub

# Download latest version
dataset_path = kagglehub.dataset_download("sachins8201/gotham")
print("Path to dataset files:", dataset_path)

dataset_path = Path(dataset_path)

# Find merged_dataset.csv anywhere inside the downloaded folder
csv_candidates = list(dataset_path.rglob("merged_dataset.csv"))
if not csv_candidates:
    raise FileNotFoundError(
        f"Could not find merged_dataset.csv under: {dataset_path}\n"
        f"Files found (sample): {[p.name for p in list(dataset_path.rglob('*'))[:20]]}"
    )

csv_path = csv_candidates[0]
print("Using CSV:", csv_path)


# =========================================================
# 2) Load data
# =========================================================
df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print(df.head(3))
print("\nLabel counts (original):")
print(df["label"].value_counts().head(20))

# =========================================================
# 3) (Optional but recommended) Balance by undersampling
#    (matches your idea: reduce big classes to median size)
# =========================================================
label_counts = df["label"].value_counts()
target_per_class = int(label_counts.median())

balanced_parts = []
for lbl, cnt in label_counts.items():
    part = df[df["label"] == lbl]
    if cnt > target_per_class:
        part = part.sample(n=target_per_class, random_state=SEED)
    balanced_parts.append(part)

df = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
del balanced_parts
gc.collect()

print("Label counts (balanced):")
print(df["label"].value_counts())
print("Balanced shape:", df.shape)


# =========================================================
# 4) Feature engineering (Colab-safe version)
#    - drops checksums
#    - time features from frame.time
#    - tcp flag bits
#    - frequency encoding for IP/MAC + some categoricals
# =========================================================

# Drop columns if present
df.drop(columns=["ip.checksum", "tcp.checksum"], errors="ignore", inplace=True)

# Time features
if "frame.time" in df.columns:
    df["frame.time"] = pd.to_datetime(df["frame.time"], errors="coerce")
    df["hour"]   = df["frame.time"].dt.hour.fillna(-1).astype("int16")
    df["minute"] = df["frame.time"].dt.minute.fillna(-1).astype("int16")
    df["second"] = df["frame.time"].dt.second.fillna(-1).astype("int16")
    df.drop(columns=["frame.time"], inplace=True)

# Protocol indicators
if "tcp.srcport" in df.columns:
    df["is_tcp"] = df["tcp.srcport"].notnull().astype("int8")
else:
    df["is_tcp"] = 0

if "udp.srcport" in df.columns:
    df["is_udp"] = df["udp.srcport"].notnull().astype("int8")
else:
    df["is_udp"] = 0

# Safe numeric fill for some columns
null_fill_cols = [
    "ip.tos","tcp.srcport","tcp.dstport","tcp.options",
    "tcp.pdu.size","udp.srcport","udp.dstport"
]
for col in null_fill_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Packet size features
if "frame.len" in df.columns:
    df["total_bytes"] = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0)
else:
    df["total_bytes"] = 0

if "tcp.window_size_value" in df.columns:
    df["tcp.window_size_value"] = pd.to_numeric(df["tcp.window_size_value"], errors="coerce").fillna(0)

if "tcp.window_size_scalefactor" in df.columns:
    df["tcp.window_size_scalefactor"] = pd.to_numeric(df["tcp.window_size_scalefactor"], errors="coerce").fillna(0)

if "tcp.window_size_value" in df.columns and "tcp.window_size_scalefactor" in df.columns:
    df["src_dst_bytes_diff"] = df["tcp.window_size_value"] - df["tcp.window_size_scalefactor"]

# TCP flags -> syn/ack bits
if "tcp.flags" in df.columns:
    flags_num = pd.to_numeric(df["tcp.flags"], errors="coerce").fillna(0).astype("int64")
    df["syn_flag"] = ((flags_num & 0x02) > 0).astype("int8")
    df["ack_flag"] = ((flags_num & 0x10) > 0).astype("int8")
    df.drop(columns=["tcp.flags"], inplace=True)

# Frequency encoding for high-cardinality columns
freq_cols = [c for c in ["ip.src","ip.dst","eth.src","eth.dst"] if c in df.columns]
for col in freq_cols:
    freq_map = df[col].value_counts()
    df[col + "_freq"] = df[col].map(freq_map).fillna(0).astype("float32")
    df.drop(columns=[col], inplace=True)

# Frequency encoding for other categoricals
cat_cols = [c for c in ["frame.protocols","ip.flags"] if c in df.columns]
for col in cat_cols:
    freq_map = df[col].value_counts()
    df[col + "_freq"] = df[col].map(freq_map).fillna(0).astype("float32")
    df.drop(columns=[col], inplace=True)

# Convert all remaining non-numeric columns (except label) using frequency encoding
# (so TabNet receives numeric matrix only)
for col in df.columns:
    if col == "label":
        continue
    if df[col].dtype == "object":
        freq_map = df[col].value_counts()
        df[col + "_freq2"] = df[col].map(freq_map).fillna(0).astype("float32")
        df.drop(columns=[col], inplace=True)

# Fill remaining NaNs
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print("Final columns:", len(df.columns))
print("Any NaNs?", df.isna().sum().sum())
df.head(3)


# =========================================================
# 5) Label encoding + train/test split + scaling
# =========================================================
le = LabelEncoder()
y = le.fit_transform(df["label"].astype(str))

X_df = df.drop(columns=["label"])
feature_names = X_df.columns.tolist()

X = X_df.astype("float32").to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Num classes:", len(le.classes_))
print("Classes:", list(le.classes_)[:10], "..." if len(le.classes_) > 10 else "")


# =========================================================
# 6) Train TabNet
# =========================================================
tabnet_clf = TabNetClassifier(
    n_d=32,
    n_a=32,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type="sparsemax",
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 10, "gamma": 0.9},
    verbose=1
)

tabnet_clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=20,
    patience=5,
    batch_size=512,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)


# =========================================================
# 7) Evaluation (report + confusion matrix)
# =========================================================
y_pred = tabnet_clf.predict(X_test)
y_prob = tabnet_clf.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Accuracy: {acc:.4f}")

prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_weight  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
hloss      = hamming_loss(y_test, y_pred)
mcc        = matthews_corrcoef(y_test, y_pred)
kappa      = cohen_kappa_score(y_test, y_pred)

print("\nSummary:")
print(f"Macro Precision: {prec_macro:.4f}")
print(f"Macro Recall   : {rec_macro:.4f}")
print(f"Macro F1       : {f1_macro:.4f}")
print(f"Weighted F1    : {f1_weight:.4f}")
print(f"Hamming Loss   : {hloss:.6f}")
print(f"MCC            : {mcc:.4f}")
print(f"Cohen Kappa    : {kappa:.4f}")

plt.figure(figsize=(8,6))
if _HAS_SNS:
    sns.heatmap(cm, annot=False, fmt="d", cbar=True)
    plt.title("Confusion Matrix")
else:
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()







---------

# ====================================================================================
# HIFN + TRANSFORMER 

!pip -q install kagglehub torch torchvision torchaudio scikit-learn seaborn

import os, gc, random, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
CONFIG = {
    # Dataset
    "kagglehub_dataset": "sachins8201/gotham",
    "target_col": "label",         # expected column name in merged_dataset.csv
    "csv_name": "merged_dataset.csv",

    # Optional balancing
    "median_undersample": True,    # like earlier workflow
    "max_rows": None,              # set e.g. 200000 for faster testing

    # HIFN
    "hifn_dims": [256, 128, 64],
    "beta_init": 0.01,

    # Transformer
    "use_transformer": True,
    "tf_seq_len": 8,
    "tf_d_model": 128,
    "tf_nhead": 8,
    "tf_num_layers": 2,
    "tf_ff_dim": 256,
    "tf_dropout": 0.1,

    # Training
    "batch_size": 512,
    "epochs": 30,                  # raise to 100-300 if you want
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "test_size": 0.2,

    # Loss weights (USED, differentiable)
    "lambda_beta": 1e-4,           # weights KL term (with betas)
    "lambda_H": 1e-3,              # weights entropy budget penalty

    # Regularization
    "dropout": 0.1,
    "grad_clip": 1.0,

    # Use class weights in CE (recommended for imbalanced)
    "use_class_weights": True,
}

# ====================================================================================
# 1) Download Gotham dataset with kagglehub
# ====================================================================================
import kagglehub

dataset_path = kagglehub.dataset_download(CONFIG["kagglehub_dataset"])
dataset_path = Path(dataset_path)
print("Dataset path:", dataset_path)

csv_candidates = list(dataset_path.rglob(CONFIG["csv_name"]))
if not csv_candidates:
    raise FileNotFoundError(
        f"Could not find {CONFIG['csv_name']} under {dataset_path}\n"
        f"Sample files: {[p.name for p in list(dataset_path.rglob('*'))[:30]]}"
    )

csv_path = csv_candidates[0]
print("Using CSV:", csv_path)

df = pd.read_csv(csv_path)
print("Original shape:", df.shape)

if CONFIG["max_rows"] is not None and len(df) > CONFIG["max_rows"]:
    df = df.sample(CONFIG["max_rows"], random_state=42).reset_index(drop=True)
    print("Downsampled for speed:", df.shape)

# ====================================================================================
# 2) Tabular preprocessing (robust)
# ====================================================================================
def preprocess_gotham(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()[:30]} ...")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop known checksum columns if present
    X.drop(columns=["ip.checksum", "tcp.checksum"], errors="ignore", inplace=True)

    # Time features if frame.time exists
    if "frame.time" in X.columns:
        X["frame.time"] = pd.to_datetime(X["frame.time"], errors="coerce")
        X["hour"]   = X["frame.time"].dt.hour.fillna(-1).astype("int16")
        X["minute"] = X["frame.time"].dt.minute.fillna(-1).astype("int16")
        X["second"] = X["frame.time"].dt.second.fillna(-1).astype("int16")
        X.drop(columns=["frame.time"], inplace=True)

    # Protocol indicators
    if "tcp.srcport" in X.columns:
        X["is_tcp"] = X["tcp.srcport"].notnull().astype("int8")
    else:
        X["is_tcp"] = 0

    if "udp.srcport" in X.columns:
        X["is_udp"] = X["udp.srcport"].notnull().astype("int8")
    else:
        X["is_udp"] = 0

    # Fill numeric-ish cols
    null_fill_cols = [
        "ip.tos","tcp.srcport","tcp.dstport","tcp.options",
        "tcp.pdu.size","udp.srcport","udp.dstport",
        "frame.len","tcp.window_size_value","tcp.window_size_scalefactor"
    ]
    for col in null_fill_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Packet size features
    if "frame.len" in X.columns:
        X["total_bytes"] = pd.to_numeric(X["frame.len"], errors="coerce").fillna(0)
    else:
        X["total_bytes"] = 0

    # Difference feature if both exist
    if "tcp.window_size_value" in X.columns and "tcp.window_size_scalefactor" in X.columns:
        X["src_dst_bytes_diff"] = X["tcp.window_size_value"] - X["tcp.window_size_scalefactor"]

    # TCP flags -> syn/ack bits
    if "tcp.flags" in X.columns:
        flags_num = pd.to_numeric(X["tcp.flags"], errors="coerce").fillna(0).astype("int64")
        X["syn_flag"] = ((flags_num & 0x02) > 0).astype("int8")
        X["ack_flag"] = ((flags_num & 0x10) > 0).astype("int8")
        X.drop(columns=["tcp.flags"], inplace=True)

    # Frequency encoding for known high-cardinality columns (if present)
    freq_cols = [c for c in ["ip.src","ip.dst","eth.src","eth.dst"] if c in X.columns]
    for col in freq_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Frequency encoding for some categoricals (if present)
    cat_cols = [c for c in ["frame.protocols","ip.flags"] if c in X.columns]
    for col in cat_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Convert ALL remaining object columns via frequency encoding
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for col in obj_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq2"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Final numeric coercion + NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

X_df, y_raw = preprocess_gotham(df, CONFIG["target_col"])
print("After preprocessing:", X_df.shape)
print("Any NaNs:", X_df.isna().sum().sum())

# Optional: median undersampling (like earlier)
if CONFIG["median_undersample"]:
    tmp = pd.concat([X_df, y_raw.rename("label")], axis=1)
    counts = tmp["label"].value_counts()
    target_per_class = int(counts.median())
    parts = []
    for lbl, cnt in counts.items():
        part = tmp[tmp["label"] == lbl]
        if cnt > target_per_class:
            part = part.sample(n=target_per_class, random_state=42)
        parts.append(part)
    tmp = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    X_df = tmp.drop(columns=["label"])
    y_raw = tmp["label"]
    print("After median undersampling:", X_df.shape)
    print("Label counts:", y_raw.value_counts().head(20))
    del tmp, parts
    gc.collect()

# Label encode
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))
class_names = le.classes_.tolist()
num_classes = len(class_names)
print("Classes:", num_classes, class_names[:10], "..." if num_classes > 10 else "")

# Numpy matrices + scaling
X = X_df.astype("float32").to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    random_state=42,
    stratify=y
)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

input_dim = X_train.shape[1]
print("Input dim:", input_dim)

# ====================================================================================
# 3) YOUR MODEL (Fixed to keep KL/entropy differentiable)
# ====================================================================================

class InformationBottleneckLayer(nn.Module):
    """
    IB Layer with learnable:
    - β (retention)      -> used in loss (differentiable)
    - B (entropy budget) -> used in entropy penalty (differentiable)
    - c (compression)
    - static gates g
    """

    def __init__(self, input_dim: int, output_dim: int, initial_beta: float = 0.01, use_stochastic: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_stochastic = use_stochastic

        self.encoder_mean = nn.Linear(input_dim, output_dim)
        self.encoder_logvar = nn.Linear(input_dim, output_dim)

        nn.init.xavier_normal_(self.encoder_mean.weight, gain=0.5)
        nn.init.constant_(self.encoder_mean.bias, 0.0)
        nn.init.constant_(self.encoder_logvar.weight, 0.0)
        nn.init.constant_(self.encoder_logvar.bias, -3.0)

        # Learnable β (log space)
        self.log_beta = nn.Parameter(torch.log(torch.tensor([initial_beta], dtype=torch.float32)))

        # Learnable entropy budget B (log space)
        self.log_entropy_budget = nn.Parameter(
            torch.log(torch.tensor([float(output_dim) * 2.0], dtype=torch.float32))
        )

        # Learnable compression ratio c ∈ [0.3, 1.0]
        self.logit_compression = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # Static gates g (logits)
        self.info_gates_logit = nn.Parameter(torch.ones(output_dim, dtype=torch.float32) * 2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_entropy_raw(self, logvar: torch.Tensor) -> torch.Tensor:
        # H = 0.5 * Σ (logvar + log(2πe)) ; return scalar tensor
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        entropy_per_dim = 0.5 * (logvar + np.log(2 * np.pi * np.e))
        return entropy_per_dim.mean(dim=0).sum()

    def forward(self, x: torch.Tensor):
        mu = self.encoder_mean(x)
        logvar = torch.clamp(self.encoder_logvar(x), min=-10.0, max=2.0)

        info_gates = torch.sigmoid(self.info_gates_logit)
        gated_mu = mu * info_gates

        if self.use_stochastic and self.training:
            z = self.reparameterize(gated_mu, logvar)
        else:
            z = gated_mu

        compression = torch.sigmoid(self.logit_compression)
        compression_scaled = 0.3 + 0.7 * compression
        z = z * compression_scaled

        # KL(q(z|x) || N(0,I))  (scalar tensor)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl, min=0.0, max=10.0)

        # Entropy (scalar tensor)
        H_raw = self.compute_entropy_raw(logvar)

        # Budget B (scalar tensor, positive)
        B = torch.exp(self.log_entropy_budget)

        # Penalty (scalar tensor) - uses RAW entropy for meaningful budget
        H_pen = F.relu(H_raw - B)

        # β value (scalar tensor)
        beta = torch.exp(self.log_beta)

        # Return z and tensor-metrics (for differentiable loss)
        metrics = {
            "kl": kl,
            "H_raw": H_raw,
            "H_pen": H_pen,
            "beta": beta,
            "B": B,
            "compression": compression_scaled,
            "gate_mean": info_gates.mean(),
        }
        return z, metrics

    def get_gate_importance(self) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.info_gates_logit).cpu().numpy()


class HIFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        beta_init: float = 0.01,
        dropout: float = 0.1,
        use_transformer: bool = True,
        tf_seq_len: int = 8,
        tf_d_model: int = 128,
        tf_nhead: int = 8,
        tf_num_layers: int = 2,
        tf_ff_dim: int = 256,
        tf_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(
                InformationBottleneckLayer(dims[i], dims[i+1], initial_beta=beta_init, use_stochastic=True)
            )
            self.dropouts.append(nn.Dropout(dropout))

        # Global β multiplier
        self.log_global_beta = nn.Parameter(torch.log(torch.tensor([beta_init], dtype=torch.float32)))

        # Transformer
        self.use_transformer = use_transformer
        self.tf_seq_len = tf_seq_len
        self.tf_d_model = tf_d_model

        if self.use_transformer:
            last_dim = hidden_dims[-1]
            self.tokenizer = nn.Linear(last_dim, tf_seq_len * tf_d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, tf_seq_len, tf_d_model))

            enc_layer = nn.TransformerEncoderLayer(
                d_model=tf_d_model,
                nhead=tf_nhead,
                dim_feedforward=tf_ff_dim,
                dropout=tf_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_num_layers)
            self.tf_norm = nn.LayerNorm(tf_d_model)
            classifier_in = tf_d_model
        else:
            classifier_in = hidden_dims[-1]

        self.classifier = nn.Linear(classifier_in, num_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

        print("\n" + "="*80)
        print("MODEL")
        print("="*80)
        print(f"Input dim: {input_dim}")
        print(f"Hidden: {' -> '.join(map(str, hidden_dims))}")
        print(f"Transformer: {self.use_transformer}")
        if self.use_transformer:
            print(f"  seq_len={tf_seq_len}, d_model={tf_d_model}, heads={tf_nhead}, layers={tf_num_layers}")
        print(f"Output classes: {num_classes}")
        print("="*80)

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        layer_metrics = []

        for layer, drop in zip(self.layers, self.dropouts):
            x, m = layer(x)
            x = F.relu(x)
            x = drop(x)
            if self.training:
                x = torch.clamp(x, min=-10.0, max=10.0)
            layer_metrics.append(m)

        if self.use_transformer:
            Bsz = x.size(0)
            tokens = self.tokenizer(x).view(Bsz, self.tf_seq_len, self.tf_d_model)
            tokens = tokens + self.pos_emb
            tokens = self.transformer(tokens)
            tokens = self.tf_norm(tokens)
            x = tokens.mean(dim=1)

        logits = self.classifier(x)
        if return_metrics:
            return logits, layer_metrics
        return logits

    def compute_information_loss(self, layer_metrics: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # kl_loss: global_beta * Σ (beta_l * KL_l)
        # h_loss:  Σ H_pen_l
        kl_sum = torch.zeros((), device=layer_metrics[0]["kl"].device)
        h_sum  = torch.zeros((), device=layer_metrics[0]["kl"].device)

        for i, m in enumerate(layer_metrics):
            beta_i = torch.exp(self.layers[i].log_beta)
            kl_sum = kl_sum + beta_i * m["kl"]
            h_sum  = h_sum + m["H_pen"]

        global_beta = torch.exp(self.log_global_beta)
        kl_loss = global_beta * kl_sum
        return kl_loss, h_sum


# ====================================================================================
# 4) Trainer
# ====================================================================================
class HIFNTrainer:
    def __init__(
        self,
        model: HIFN,
        learning_rate: float,
        weight_decay: float,
        lambda_beta: float,
        lambda_H: float,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_beta = float(lambda_beta)
        self.lambda_H = float(lambda_H)
        self.class_weights = class_weights

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "kl_loss": [], "h_loss": [],
            "lr": []
        }

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        total_kl, total_h = 0.0, 0.0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad()

            logits, metrics = self.model(xb, return_metrics=True)

            ce = F.cross_entropy(logits, yb, weight=self.class_weights)
            kl_loss, h_loss = self.model.compute_information_loss(metrics)

            loss = ce + self.lambda_beta * kl_loss + self.lambda_H * h_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=CONFIG["grad_clip"])
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_kl   += float(kl_loss.detach().cpu().item())
            total_h    += float(h_loss.detach().cpu().item())

            pred = logits.argmax(dim=1)
            total_correct += int((pred == yb).sum().item())
            total_n += int(yb.size(0))

        avg_loss = total_loss / max(len(loader), 1)
        acc = 100.0 * total_correct / max(total_n, 1)
        return avg_loss, acc, total_kl / max(len(loader), 1), total_h / max(len(loader), 1)

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb, weight=self.class_weights)
                total_loss += float(loss.detach().cpu().item())
                pred = logits.argmax(dim=1)
                total_correct += int((pred == yb).sum().item())
                total_n += int(yb.size(0))

        avg_loss = total_loss / max(len(loader), 1)
        acc = 100.0 * total_correct / max(total_n, 1)
        return avg_loss, acc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_acc = 0.0
        best_state = None

        print("\n" + "="*80)
        print("TRAIN")
        print("="*80)
        print(f"epochs={epochs} batch={train_loader.batch_size} lambda_beta={self.lambda_beta} lambda_H={self.lambda_H}")
        print("="*80)

        for ep in range(epochs):
            tr_loss, tr_acc, tr_kl, tr_h = self.train_epoch(train_loader)
            va_loss, va_acc = self.validate(val_loader)

            self.scheduler.step(va_acc)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history["kl_loss"].append(tr_kl)
            self.history["h_loss"].append(tr_h)
            self.history["lr"].append(lr)

            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (ep + 1) % 5 == 0:
                g_beta = float(torch.exp(self.model.log_global_beta).detach().cpu().item())
                print(
                    f"Epoch {ep+1:3d}/{epochs} | "
                    f"TrainAcc {tr_acc:6.2f}% ValAcc {va_acc:6.2f}% | "
                    f"TrainLoss {tr_loss:.4f} ValLoss {va_loss:.4f} | "
                    f"KL {tr_kl:.4f} H {tr_h:.4f} | gβ {g_beta:.4f} | LR {lr:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print("\nBest ValAcc:", best_acc)
        return self.history

    def predict(self, loader: DataLoader):
        self.model.eval()
        preds, trues = [], []
        probs = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                pred = np.argmax(p, axis=1)
                probs.append(p)
                preds.extend(pred.tolist())
                trues.extend(yb.numpy().tolist())

        probs = np.vstack(probs) if len(probs) else None
        return np.array(preds), np.array(trues), probs


# ====================================================================================
# 5) Class weights (optional)
# ====================================================================================
class_weights = None
if CONFIG["use_class_weights"]:
    counts = np.bincount(y_train)
    w = (counts.sum() / (counts + 1e-9))
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("Class weights:", class_weights.detach().cpu().numpy())

# ====================================================================================
# 6) Build + Train model
# ====================================================================================
model = HIFN(
    input_dim=input_dim,
    hidden_dims=CONFIG["hifn_dims"],
    num_classes=num_classes,
    beta_init=CONFIG["beta_init"],
    dropout=CONFIG["dropout"],
    use_transformer=CONFIG["use_transformer"],
    tf_seq_len=CONFIG["tf_seq_len"],
    tf_d_model=CONFIG["tf_d_model"],
    tf_nhead=CONFIG["tf_nhead"],
    tf_num_layers=CONFIG["tf_num_layers"],
    tf_ff_dim=CONFIG["tf_ff_dim"],
    tf_dropout=CONFIG["tf_dropout"],
)

trainer = HIFNTrainer(
    model=model,
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    lambda_beta=CONFIG["lambda_beta"],
    lambda_H=CONFIG["lambda_H"],
    device=device,
    class_weights=class_weights
)

history = trainer.fit(train_loader, test_loader, epochs=CONFIG["epochs"])

# ====================================================================================
# 7) Evaluation
# ====================================================================================
y_pred, y_true, y_prob = trainer.predict(test_loader)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print("\n" + "="*80)
print("EVALUATION (TEST)")
print("="*80)
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
cmn = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

plt.figure(figsize=(10, 8))
sns.heatmap(cmn, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Multi-class ROC-AUC (OVR) if probabilities available and > 1 class
if y_prob is not None and num_classes > 1:
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        print(f"ROC-AUC (macro, OVR): {roc_auc:.4f}")
    except Exception as e:
        print("ROC-AUC skipped:", e)

# ====================================================================================
# 8) Plots: training curves
# ====================================================================================
plt.figure(figsize=(12, 5))
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acc (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history["kl_loss"], label="KL term")
plt.plot(history["h_loss"], label="Entropy penalty term")
plt.title("IB Regularization Terms")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ====================================================================================
# 9) Save artifacts
# ====================================================================================
save_obj = {
    "model_state_dict": model.state_dict(),
    "config": CONFIG,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "label_classes": class_names,
    "input_dim": input_dim,
    "history": history
}

torch.save(save_obj, "hifn_gotham_tabular.pt")
print("\nSaved: hifn_gotham_tabular.pt")





-----------------

# ====================================================================================
# HIFN + TRANSFORMER (Tabular Gotham) — FULL COLAB CODE (NO PLACEHOLDERS)
# - Removes "Unknown" label BEFORE balancing/training
# - Full metrics (MCC, Kappa, Balanced Acc, LogLoss, ROC-AUC, PR-AUC, etc.)
# - Plots like your figure (2x2): Accuracy, CE, KL, Entropy RAW+CLAMP
# - Confusion matrices with numbers (counts + normalized)
# - XAI: SHAP (KernelExplainer) + LIME (tabular)
# ====================================================================================

!pip -q install kagglehub scikit-learn seaborn shap lime

import os, gc, random, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    hamming_loss, log_loss,
    roc_auc_score, average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lime.lime_tabular import LimeTabularExplainer

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ====================================================================================
# CONFIG
# ====================================================================================
CONFIG = {
    "kagglehub_dataset": "sachins8201/gotham",
    "csv_name": "merged_dataset.csv",
    "target_col": "label",

    # remove label(s)
    "drop_unknown_label": True,    # <-- YOU ASKED THIS
    "unknown_values": {"unknown", "unk", "nan", ""},

    # optional speed controls
    "max_rows": None,              # e.g. 200000 for faster test; None uses all
    "median_undersample": True,    # your balancing approach

    # model
    "hifn_dims": [256, 128, 64],
    "beta_init": 0.01,

    # transformer
    "use_transformer": True,
    "tf_seq_len": 8,
    "tf_d_model": 128,
    "tf_nhead": 8,
    "tf_num_layers": 2,
    "tf_ff_dim": 256,
    "tf_dropout": 0.1,

    # training
    "batch_size": 512,
    "epochs": 30,                  # increase to 100-300 if you want
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "test_size": 0.2,
    "grad_clip": 1.0,
    "dropout": 0.1,

    # IB weights (USED)
    "lambda_beta": 1e-4,
    "lambda_H": 1e-3,

    # class weights in CE
    "use_class_weights": True,

    # XAI settings (controlled for runtime, but NO placeholders)
    "xai_num_test_samples": 300,
    "shap_background_size": 60,
    "shap_explain_size": 120,
    "shap_nsamples": 200,
    "lime_num_instances": 5,
    "lime_num_features": 15,
}

# ====================================================================================
# 1) Download dataset with kagglehub + load CSV
# ====================================================================================
import kagglehub

dataset_path = Path(kagglehub.dataset_download(CONFIG["kagglehub_dataset"]))
print("Dataset path:", dataset_path)

csv_candidates = list(dataset_path.rglob(CONFIG["csv_name"]))
if not csv_candidates:
    raise FileNotFoundError(
        f"Could not find {CONFIG['csv_name']} under {dataset_path}\n"
        f"Sample files: {[p.name for p in list(dataset_path.rglob('*'))[:30]]}"
    )

csv_path = csv_candidates[0]
print("Using CSV:", csv_path)

df = pd.read_csv(csv_path)
print("Original shape:", df.shape)

if CONFIG["max_rows"] is not None and len(df) > CONFIG["max_rows"]:
    df = df.sample(CONFIG["max_rows"], random_state=42).reset_index(drop=True)
    print("Downsampled for speed:", df.shape)

# ====================================================================================
# 2) REMOVE "Unknown" label BEFORE ANYTHING (as you requested)
# ====================================================================================
target_col = CONFIG["target_col"]

df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col].notna()].copy()

if CONFIG["drop_unknown_label"]:
    lower = df[target_col].str.lower()
    df = df[~lower.isin(CONFIG["unknown_values"])].copy()
    # also remove literal "Unknown" with extra spaces/case
    df = df[df[target_col].str.lower() != "unknown"].copy()

print("\nAfter removing Unknown/blank labels:")
print("Shape:", df.shape)
print("Unique labels:", df[target_col].nunique())
print(df[target_col].value_counts().head(30))

# ====================================================================================
# 3) Preprocessing (tabular)
# ====================================================================================
def preprocess_gotham(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop checksums
    X.drop(columns=["ip.checksum", "tcp.checksum"], errors="ignore", inplace=True)

    # frame.time -> hour/min/sec
    if "frame.time" in X.columns:
        X["frame.time"] = pd.to_datetime(X["frame.time"], errors="coerce")
        X["hour"]   = X["frame.time"].dt.hour.fillna(-1).astype("int16")
        X["minute"] = X["frame.time"].dt.minute.fillna(-1).astype("int16")
        X["second"] = X["frame.time"].dt.second.fillna(-1).astype("int16")
        X.drop(columns=["frame.time"], inplace=True)

    # Protocol indicators
    X["is_tcp"] = X["tcp.srcport"].notnull().astype("int8") if "tcp.srcport" in X.columns else 0
    X["is_udp"] = X["udp.srcport"].notnull().astype("int8") if "udp.srcport" in X.columns else 0

    # Coerce numeric & fill NaNs for known columns
    fill_cols = [
        "ip.tos", "tcp.srcport", "tcp.dstport", "tcp.options",
        "tcp.pdu.size", "udp.srcport", "udp.dstport",
        "frame.len", "tcp.window_size_value", "tcp.window_size_scalefactor"
    ]
    for col in fill_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # total_bytes
    if "frame.len" in X.columns:
        X["total_bytes"] = pd.to_numeric(X["frame.len"], errors="coerce").fillna(0)
    else:
        X["total_bytes"] = 0

    # diff feature
    if "tcp.window_size_value" in X.columns and "tcp.window_size_scalefactor" in X.columns:
        X["src_dst_bytes_diff"] = X["tcp.window_size_value"] - X["tcp.window_size_scalefactor"]

    # tcp.flags -> syn/ack
    if "tcp.flags" in X.columns:
        flags = pd.to_numeric(X["tcp.flags"], errors="coerce").fillna(0).astype("int64")
        X["syn_flag"] = ((flags & 0x02) > 0).astype("int8")
        X["ack_flag"] = ((flags & 0x10) > 0).astype("int8")
        X.drop(columns=["tcp.flags"], inplace=True)

    # Frequency encode high-cardinality
    freq_cols = [c for c in ["ip.src", "ip.dst", "eth.src", "eth.dst"] if c in X.columns]
    for col in freq_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Frequency encode some categoricals
    cat_cols = [c for c in ["frame.protocols", "ip.flags"] if c in X.columns]
    for col in cat_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Any remaining object -> frequency encoding
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for col in obj_cols:
        freq_map = X[col].value_counts()
        X[col + "_freq2"] = X[col].map(freq_map).fillna(0).astype("float32")
        X.drop(columns=[col], inplace=True)

    # Final numeric + NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

X_df, y_raw = preprocess_gotham(df, target_col)
feature_names = X_df.columns.tolist()

print("\nAfter preprocessing:", X_df.shape, "| NaNs:", X_df.isna().sum().sum())
print("Label counts (raw, top 30):\n", y_raw.value_counts().head(30))

# ====================================================================================
# 4) Median undersampling (optional) — AFTER dropping Unknown
# ====================================================================================
if CONFIG["median_undersample"]:
    tmp = pd.concat([X_df, y_raw.rename("label")], axis=1)
    counts = tmp["label"].value_counts()
    target_per_class = int(counts.median())

    parts = []
    for lbl, cnt in counts.items():
        part = tmp[tmp["label"] == lbl]
        if cnt > target_per_class:
            part = part.sample(n=target_per_class, random_state=42)
        parts.append(part)

    tmp = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    X_df = tmp.drop(columns=["label"])
    y_raw = tmp["label"]
    feature_names = X_df.columns.tolist()

    print("\nLabel counts (balanced):")
    print(y_raw.value_counts())
    print("Balanced shape:", tmp.shape)

    del tmp, parts
    gc.collect()

# ====================================================================================
# 5) Encode labels + scale
# ====================================================================================
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))
class_names = le.classes_.tolist()
num_classes = len(class_names)

X = X_df.astype("float32").to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG["test_size"], random_state=42, stratify=y
)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

input_dim = X_train.shape[1]
print("\nInput dim:", input_dim)
print("Classes:", num_classes)
print(class_names)

# ====================================================================================
# 6) HIFN + Transformer (Differentiable KL + Entropy Budget)
# ====================================================================================
class InformationBottleneckLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, initial_beta: float = 0.01, use_stochastic: bool = True):
        super().__init__()
        self.use_stochastic = use_stochastic

        self.encoder_mean = nn.Linear(input_dim, output_dim)
        self.encoder_logvar = nn.Linear(input_dim, output_dim)

        nn.init.xavier_normal_(self.encoder_mean.weight, gain=0.5)
        nn.init.constant_(self.encoder_mean.bias, 0.0)
        nn.init.constant_(self.encoder_logvar.weight, 0.0)
        nn.init.constant_(self.encoder_logvar.bias, -3.0)

        self.log_beta = nn.Parameter(torch.log(torch.tensor([initial_beta], dtype=torch.float32)))
        self.log_entropy_budget = nn.Parameter(torch.log(torch.tensor([float(output_dim) * 2.0], dtype=torch.float32)))
        self.logit_compression = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.info_gates_logit = nn.Parameter(torch.ones(output_dim, dtype=torch.float32) * 2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_entropy_raw(self, logvar: torch.Tensor) -> torch.Tensor:
        # H = 0.5 * Σ (logvar + log(2πe)) ; scalar tensor
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        entropy_per_dim = 0.5 * (logvar + np.log(2 * np.pi * np.e))
        return entropy_per_dim.mean(dim=0).sum()

    def forward(self, x: torch.Tensor):
        mu = self.encoder_mean(x)
        logvar = torch.clamp(self.encoder_logvar(x), min=-10.0, max=2.0)

        gates = torch.sigmoid(self.info_gates_logit)
        gated_mu = mu * gates

        if self.use_stochastic and self.training:
            z = self.reparameterize(gated_mu, logvar)
        else:
            z = gated_mu

        comp = torch.sigmoid(self.logit_compression)
        comp_scaled = 0.3 + 0.7 * comp
        z = z * comp_scaled

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl, min=0.0, max=10.0)

        H_raw = self.compute_entropy_raw(logvar)
        H_clamp = torch.clamp(H_raw, min=-50.0, max=50.0)   # for plotting only

        B = torch.exp(self.log_entropy_budget)
        H_pen = F.relu(H_raw - B)  # uses RAW entropy (meaningful)

        beta = torch.exp(self.log_beta)

        metrics = {
            "kl": kl,
            "H_raw": H_raw,
            "H_clamp": H_clamp,
            "H_pen": H_pen,
            "beta": beta,
            "B": B,
            "compression": comp_scaled,
            "gate_mean": gates.mean(),
            "gate_std": gates.std(unbiased=False),
        }
        return z, metrics


class HIFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        beta_init: float,
        dropout: float,
        use_transformer: bool,
        tf_seq_len: int,
        tf_d_model: int,
        tf_nhead: int,
        tf_num_layers: int,
        tf_ff_dim: int,
        tf_dropout: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(InformationBottleneckLayer(dims[i], dims[i+1], initial_beta=beta_init, use_stochastic=True))
            self.dropouts.append(nn.Dropout(dropout))

        self.log_global_beta = nn.Parameter(torch.log(torch.tensor([beta_init], dtype=torch.float32)))

        self.use_transformer = use_transformer
        self.tf_seq_len = tf_seq_len
        self.tf_d_model = tf_d_model

        if self.use_transformer:
            last_dim = hidden_dims[-1]
            self.tokenizer = nn.Linear(last_dim, tf_seq_len * tf_d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, tf_seq_len, tf_d_model))

            enc_layer = nn.TransformerEncoderLayer(
                d_model=tf_d_model,
                nhead=tf_nhead,
                dim_feedforward=tf_ff_dim,
                dropout=tf_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_num_layers)
            self.tf_norm = nn.LayerNorm(tf_d_model)
            classifier_in = tf_d_model
        else:
            classifier_in = hidden_dims[-1]

        self.classifier = nn.Linear(classifier_in, num_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

        print("\n" + "="*80)
        print("MODEL")
        print("="*80)
        print(f"Input: {input_dim}")
        print(f"Hidden: {' -> '.join(map(str, hidden_dims))}")
        print(f"Transformer: {use_transformer}")
        if use_transformer:
            print(f"  seq_len={tf_seq_len} d_model={tf_d_model} heads={tf_nhead} layers={tf_num_layers}")
        print(f"Classes: {num_classes}")
        print("="*80)

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        layer_metrics = []
        for layer, drop in zip(self.layers, self.dropouts):
            x, m = layer(x)
            x = F.relu(x)
            x = drop(x)
            if self.training:
                x = torch.clamp(x, min=-10.0, max=10.0)
            layer_metrics.append(m)

        if self.use_transformer:
            Bsz = x.size(0)
            tokens = self.tokenizer(x).view(Bsz, self.tf_seq_len, self.tf_d_model)
            tokens = tokens + self.pos_emb
            tokens = self.transformer(tokens)
            tokens = self.tf_norm(tokens)
            x = tokens.mean(dim=1)

        logits = self.classifier(x)
        if return_metrics:
            return logits, layer_metrics
        return logits

    def compute_information_loss(self, layer_metrics):
        # kl_loss: global_beta * Σ (beta_l * KL_l)
        # h_loss:  Σ H_pen_l
        kl_sum = torch.zeros((), device=layer_metrics[0]["kl"].device)
        h_sum  = torch.zeros((), device=layer_metrics[0]["kl"].device)

        for i, m in enumerate(layer_metrics):
            beta_i = torch.exp(self.layers[i].log_beta)
            kl_sum = kl_sum + beta_i * m["kl"]
            h_sum  = h_sum + m["H_pen"]

        global_beta = torch.exp(self.log_global_beta)
        kl_loss = global_beta * kl_sum
        return kl_loss, h_sum

    def info_flow_sums(self, layer_metrics):
        # For plotting/logging
        kl_sum = torch.stack([m["kl"] for m in layer_metrics]).sum()
        H_raw_sum = torch.stack([m["H_raw"] for m in layer_metrics]).sum()
        H_clamp_sum = torch.stack([m["H_clamp"] for m in layer_metrics]).sum()
        H_pen_sum = torch.stack([m["H_pen"] for m in layer_metrics]).sum()
        return kl_sum, H_raw_sum, H_clamp_sum, H_pen_sum


# ====================================================================================
# 7) Trainer (tracks ALL LOSSES + KL/Entropy sums like your plot)
# ====================================================================================
class HIFNTrainer:
    def __init__(self, model: HIFN, class_weights: torch.Tensor | None):
        self.model = model.to(device)
        self.class_weights = class_weights

        self.optimizer = Adam(self.model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)

        self.history = {
            "train_acc": [], "val_acc": [],
            "train_ce": [], "val_ce": [],
            "train_total": [],
            "train_kl_loss": [], "train_h_loss": [],
            "train_kl_sum": [],
            "train_entropy_raw_sum": [],
            "train_entropy_clamp_sum": [],
            "lr": []
        }

    def ce_loss(self, logits, yb):
        return F.cross_entropy(logits, yb, weight=self.class_weights)

    def train_epoch(self, loader):
        self.model.train()
        total_correct, total_n = 0, 0

        sum_total = 0.0
        sum_ce = 0.0
        sum_kl_loss = 0.0
        sum_h_loss = 0.0

        kl_sums = []
        Hraw_sums = []
        Hclamp_sums = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            self.optimizer.zero_grad()

            logits, layer_metrics = self.model(xb, return_metrics=True)

            ce = self.ce_loss(logits, yb)
            kl_loss, h_loss = self.model.compute_information_loss(layer_metrics)

            total_loss = ce + CONFIG["lambda_beta"] * kl_loss + CONFIG["lambda_H"] * h_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=CONFIG["grad_clip"])
            self.optimizer.step()

            sum_total += float(total_loss.detach().cpu().item())
            sum_ce += float(ce.detach().cpu().item())
            sum_kl_loss += float(kl_loss.detach().cpu().item())
            sum_h_loss += float(h_loss.detach().cpu().item())

            pred = logits.argmax(dim=1)
            total_correct += int((pred == yb).sum().item())
            total_n += int(yb.size(0))

            kl_sum, H_raw_sum, H_clamp_sum, _ = self.model.info_flow_sums(layer_metrics)
            kl_sums.append(float(kl_sum.detach().cpu().item()))
            Hraw_sums.append(float(H_raw_sum.detach().cpu().item()))
            Hclamp_sums.append(float(H_clamp_sum.detach().cpu().item()))

        avg_total = sum_total / max(len(loader), 1)
        avg_ce = sum_ce / max(len(loader), 1)
        avg_kl_loss = sum_kl_loss / max(len(loader), 1)
        avg_h_loss = sum_h_loss / max(len(loader), 1)

        acc = 100.0 * total_correct / max(total_n, 1)

        avg_kl_sum = float(np.mean(kl_sums)) if len(kl_sums) else 0.0
        avg_Hraw_sum = float(np.mean(Hraw_sums)) if len(Hraw_sums) else 0.0
        avg_Hclamp_sum = float(np.mean(Hclamp_sums)) if len(Hclamp_sums) else 0.0

        return avg_total, avg_ce, avg_kl_loss, avg_h_loss, acc, avg_kl_sum, avg_Hraw_sum, avg_Hclamp_sum

    def validate(self, loader):
        self.model.eval()
        total_ce, total_correct, total_n = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = self.model(xb)
                ce = self.ce_loss(logits, yb)
                total_ce += float(ce.detach().cpu().item())
                pred = logits.argmax(dim=1)
                total_correct += int((pred == yb).sum().item())
                total_n += int(yb.size(0))

        avg_ce = total_ce / max(len(loader), 1)
        acc = 100.0 * total_correct / max(total_n, 1)
        return avg_ce, acc

    def fit(self, train_loader, val_loader, epochs):
        best_acc = 0.0
        best_state = None

        print("\n" + "="*80)
        print("TRAIN")
        print("="*80)
        print(f"epochs={epochs} batch={train_loader.batch_size}")
        print(f"lambda_beta={CONFIG['lambda_beta']} lambda_H={CONFIG['lambda_H']}")
        print("="*80)

        for ep in range(epochs):
            tr_total, tr_ce, tr_kl_loss, tr_h_loss, tr_acc, tr_kl_sum, tr_Hraw, tr_Hclamp = self.train_epoch(train_loader)
            va_ce, va_acc = self.validate(val_loader)

            self.scheduler.step(va_acc)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_total"].append(tr_total)
            self.history["train_ce"].append(tr_ce)
            self.history["val_ce"].append(va_ce)

            self.history["train_kl_loss"].append(tr_kl_loss)
            self.history["train_h_loss"].append(tr_h_loss)

            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            self.history["train_kl_sum"].append(tr_kl_sum)
            self.history["train_entropy_raw_sum"].append(tr_Hraw)
            self.history["train_entropy_clamp_sum"].append(tr_Hclamp)

            self.history["lr"].append(lr)

            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (ep + 1) % 10 == 0:
                g_beta = float(torch.exp(self.model.log_global_beta).detach().cpu().item())
                print(
                    f"Epoch {ep+1:3d}/{epochs} | "
                    f"TrainAcc {tr_acc:6.2f}% ValAcc {va_acc:6.2f}% | "
                    f"TrainCE {tr_ce:.4f} ValCE {va_ce:.4f} | "
                    f"KLsum {tr_kl_sum:.4f} | "
                    f"Hraw {tr_Hraw:.2f} Hclamp {tr_Hclamp:.2f} | "
                    f"KLloss {tr_kl_loss:.4f} Hloss {tr_h_loss:.4f} | "
                    f"gβ {g_beta:.4f} | LR {lr:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print("\n✅ Best ValAcc:", best_acc)
        return self.history

    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []
        probs = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = self.model(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                pred = np.argmax(p, axis=1)
                probs.append(p)
                preds.extend(pred.tolist())
                trues.extend(yb.numpy().tolist())

        probs = np.vstack(probs) if len(probs) else None
        return np.array(preds), np.array(trues), probs


# ====================================================================================
# 8) Class weights (optional)
# ====================================================================================
class_weights = None
if CONFIG["use_class_weights"]:
    counts = np.bincount(y_train)
    w = (counts.sum() / (counts + 1e-9))
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("\nClass weights:", class_weights.detach().cpu().numpy())

# ====================================================================================
# 9) Train model
# ====================================================================================
model = HIFN(
    input_dim=input_dim,
    hidden_dims=CONFIG["hifn_dims"],
    num_classes=num_classes,
    beta_init=CONFIG["beta_init"],
    dropout=CONFIG["dropout"],
    use_transformer=CONFIG["use_transformer"],
    tf_seq_len=CONFIG["tf_seq_len"],
    tf_d_model=CONFIG["tf_d_model"],
    tf_nhead=CONFIG["tf_nhead"],
    tf_num_layers=CONFIG["tf_num_layers"],
    tf_ff_dim=CONFIG["tf_ff_dim"],
    tf_dropout=CONFIG["tf_dropout"],
)

trainer = HIFNTrainer(model=model, class_weights=class_weights)
history = trainer.fit(train_loader, test_loader, epochs=CONFIG["epochs"])

# ====================================================================================
# 10) Plots like your figure (2x2): Accuracy, CE, KL, Entropy RAW+CLAMP
# ====================================================================================
def plot_like_your_figure(hist: dict, save_path: str = "training_4plots.png"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Accuracy
    ax = axes[0, 0]
    ax.plot(hist["train_acc"], label="Train")
    ax.plot(hist["val_acc"], label="Validation")
    ax.set_title("Classification Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cross-Entropy
    ax = axes[0, 1]
    ax.plot(hist["train_ce"], label="Train")
    ax.plot(hist["val_ce"], label="Validation")
    ax.set_title("Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL sum
    ax = axes[1, 0]
    ax.plot(hist["train_kl_sum"], label="KL (sum across layers)")
    ax.set_title("Information Bottleneck Regularization (KL)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy RAW + CLAMP
    ax = axes[1, 1]
    ax.plot(hist["train_entropy_raw_sum"], label="Total Entropy RAW")
    ax.plot(hist["train_entropy_clamp_sum"], label="Total Entropy CLAMP")
    ax.set_title("Information Content (Entropy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ Saved:", save_path)

plot_like_your_figure(history, save_path="training_4plots.png")

# ====================================================================================
# 11) Evaluation + FULL METRICS (MCC, Kappa, ROC-AUC, PR-AUC, etc.)
# ====================================================================================
y_pred, y_true, y_prob = trainer.predict(test_loader)

acc = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)

prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
rec_micro  = recall_score(y_true, y_pred, average="micro", zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro", zero_division=0)

prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec_w  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
hloss = hamming_loss(y_true, y_pred)

ll = log_loss(y_true, y_prob, labels=list(range(num_classes))) if (y_prob is not None and num_classes > 1) else np.nan

roc_auc_macro = np.nan
pr_auc_macro  = np.nan
if y_prob is not None and num_classes > 1:
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        roc_auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc_macro = np.nan
    try:
        pr_auc_macro = average_precision_score(y_true_bin, y_prob, average="macro")
    except Exception:
        pr_auc_macro = np.nan

print("\n" + "="*80)
print("EVALUATION (TEST) — FULL METRICS")
print("="*80)
print(f"Accuracy                : {acc:.4f}")
print(f"Balanced Accuracy       : {bacc:.4f}")
print(f"Macro Precision/Recall/F1   : {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
print(f"Micro Precision/Recall/F1   : {prec_micro:.4f} / {rec_micro:.4f} / {f1_micro:.4f}")
print(f"Weighted Precision/Recall/F1: {prec_w:.4f} / {rec_w:.4f} / {f1_w:.4f}")
print(f"MCC                     : {mcc:.4f}")
print(f"Cohen's Kappa           : {kappa:.4f}")
print(f"Hamming Loss            : {hloss:.6f}")
print(f"Log Loss                : {ll:.6f}")
print(f"ROC-AUC (macro, OVR)    : {roc_auc_macro:.4f}")
print(f"PR-AUC (macro)          : {pr_auc_macro:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# ====================================================================================
# 12) Confusion matrix WITH NUMBERS (counts + normalized)
# ====================================================================================
def plot_confusion_matrices(y_true, y_pred, class_names, prefix="cm"):
    cm = confusion_matrix(y_true, y_pred)

    # --- counts matrix ---
    plt.figure(figsize=(12, 10))
    annot_counts = True if len(class_names) <= 30 else False  # avoids unreadable huge plot
    sns.heatmap(cm, annot=annot_counts, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{prefix}_counts.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ Saved:", f"{prefix}_counts.png")

    # --- normalized matrix ---
    cmn = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(12, 10))
    annot_norm = True if len(class_names) <= 30 else False
    sns.heatmap(cmn, annot=annot_norm, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{prefix}_normalized.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ Saved:", f"{prefix}_normalized.png")

plot_confusion_matrices(y_true, y_pred, class_names, prefix="confusion_matrix")

# ====================================================================================
# 13) XAI — SHAP (KernelExplainer) + LIME
# ====================================================================================
def predict_proba_np(X_np: np.ndarray) -> np.ndarray:
    X_np = X_np.astype("float32")
    xt = torch.tensor(X_np, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(xt)
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()
    return prob

# XAI pool from test
rng = np.random.RandomState(42)
n_pool = min(CONFIG["xai_num_test_samples"], X_test.shape[0])
pool_idx = rng.choice(X_test.shape[0], size=n_pool, replace=False)
X_test_pool = X_test[pool_idx]
y_test_pool = y_test[pool_idx]

# ---- SHAP KernelExplainer ----
print("\n" + "="*80)
print("XAI: SHAP (KernelExplainer)")
print("="*80)

bg_size = min(CONFIG["shap_background_size"], X_train.shape[0])
bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
X_bg = X_train[bg_idx]

explain_size = min(CONFIG["shap_explain_size"], X_test_pool.shape[0])
X_explain = X_test_pool[:explain_size]

explainer = shap.KernelExplainer(predict_proba_np, X_bg)
shap_values = explainer.shap_values(X_explain, nsamples=CONFIG["shap_nsamples"])

np.savez_compressed("shap_values_kernel.npz", *shap_values)
np.save("shap_X_explain.npy", X_explain)
print("✅ Saved: shap_values_kernel.npz, shap_X_explain.npy")

# Global importance = mean(|SHAP|) across classes
abs_means = np.zeros((X_explain.shape[1],), dtype=np.float64)
for c in range(num_classes):
    abs_means += np.mean(np.abs(shap_values[c]), axis=0)
abs_means /= num_classes

topk = min(30, len(feature_names))
top_idx = np.argsort(abs_means)[::-1][:topk]
top_features = [feature_names[i] for i in top_idx]
top_scores = abs_means[top_idx]

plt.figure(figsize=(10, 8))
plt.barh(top_features[::-1], top_scores[::-1])
plt.title("SHAP Global Feature Importance (mean |SHAP| across classes)")
plt.xlabel("mean(|SHAP|)")
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: shap_global_importance.png")

# Beeswarm for most common test-pool class
most_common_class = int(pd.Series(y_test_pool).value_counts().idxmax())
print("Beeswarm class:", most_common_class, "| name:", class_names[most_common_class])

shap.summary_plot(
    shap_values[most_common_class],
    X_explain,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_beeswarm_selected_class.png", dpi=300, bbox_inches="tight")
plt

plt.show()
print("✅ Saved: shap_beeswarm_selected_class.png")

# ---- LIME ----
print("\n" + "="*80)
print("XAI: LIME (Tabular)")
print("="*80)

lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_dir = Path("lime_reports")
lime_dir.mkdir(exist_ok=True)

num_lime = min(CONFIG["lime_num_instances"], X_test_pool.shape[0])
for i in range(num_lime):
    x_i = X_test_pool[i]
    true_i = y_test_pool[i]

    exp = lime_explainer.explain_instance(
        data_row=x_i,
        predict_fn=predict_proba_np,
        num_features=CONFIG["lime_num_features"],
        top_labels=min(3, num_classes)
    )

    html_path = lime_dir / f"lime_instance_{i}_true_{class_names[true_i]}.html"
    exp.save_to_file(str(html_path))
    print("✅ Saved:", html_path)

# ====================================================================================
# 14) Save artifacts
# ====================================================================================
save_obj = {
    "model_state_dict": model.state_dict(),
    "config": CONFIG,
    "feature_names": feature_names,
    "label_classes": class_names,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "input_dim": input_dim,
    "history": history,
    "metrics": {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_w,
        "weighted_recall": rec_w,
        "weighted_f1": f1_w,
        "mcc": mcc,
        "kappa": kappa,
        "hamming_loss": hloss,
        "log_loss": ll,
        "roc_auc_macro_ovr": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
    }
}
torch.save(save_obj, "hifn_gotham_tabular_full_no_unknown.pt")

print("\n✅ Saved model package: hifn_gotham_tabular_full_no_unknown.pt")
print("\nSaved files:")
print("- training_4plots.png")
print("- confusion_matrix_counts.png, confusion_matrix_normalized.png")
print("- shap_values_kernel.npz, shap_X_explain.npy")
print("- shap_global_importance.png, shap_beeswarm_selected_class.png")
print("- lime_reports/*.html")
print("- hifn_gotham_tabular_full_no_unknown.pt")



# ============================
# FIX SHAP SHAPES (NO RETRAIN)
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def _as_list_of_class_arrays(shap_values, num_classes_expected=None):
    """
    Normalize SHAP output to: list of arrays, each shape (N, F)
    Handles:
      - list of arrays
      - ndarray of shape (C, N, F) or (N, F, C)
    """
    if isinstance(shap_values, list):
        return shap_values

    sv = np.array(shap_values, dtype=object)
    # If sv is numeric ndarray
    if isinstance(shap_values, np.ndarray) and shap_values.dtype != object:
        if shap_values.ndim == 3:
            # guess ordering
            # case A: (C, N, F)
            if num_classes_expected is not None and shap_values.shape[0] == num_classes_expected:
                return [shap_values[c] for c in range(shap_values.shape[0])]
            # case B: (N, F, C)
            if num_classes_expected is not None and shap_values.shape[-1] == num_classes_expected:
                return [shap_values[:, :, c] for c in range(shap_values.shape[-1])]
            # fallback: treat first axis as classes
            return [shap_values[c] for c in range(shap_values.shape[0])]
        elif shap_values.ndim == 2:
            # binary or single output => wrap
            return [shap_values]
    # fallback if object array
    if sv.ndim == 1:
        return list(sv)
    raise ValueError(f"Cannot normalize shap_values of type/shape: {type(shap_values)} / {getattr(shap_values,'shape',None)}")


def _fix_feature_dim(sv_list, target_F):
    """
    Ensure each class array is shape (N, target_F).
    If F differs, crop or pad with zeros (safe for plotting global importance).
    """
    fixed = []
    for a in sv_list:
        a = np.asarray(a)
        if a.ndim != 2:
            # try squeeze
            a = np.squeeze(a)
            if a.ndim != 2:
                raise ValueError(f"SHAP class array not 2D after squeeze: shape={a.shape}")

        N, F = a.shape
        if F == target_F:
            fixed.append(a)
        elif F > target_F:
            fixed.append(a[:, :target_F])
        else:
            pad = np.zeros((N, target_F - F), dtype=a.dtype)
            fixed.append(np.concatenate([a, pad], axis=1))
    return fixed


# ---- 1) Normalize to list[class] ----
sv_list = _as_list_of_class_arrays(shap_values, num_classes_expected=num_classes)

# Sometimes KernelExplainer returns only some classes; keep what exists
print("Raw SHAP classes returned:", len(sv_list), " | expected:", num_classes)

# ---- 2) Fix feature dimension to match X_explain ----
F_target = X_explain.shape[1]
sv_list = _fix_feature_dim(sv_list, target_F=F_target)

print("Feature dim target:", F_target, " | class[0] shape:", sv_list[0].shape)

# ---- 3) Global importance (mean |SHAP|) across returned classes ----
abs_means = np.zeros((F_target,), dtype=np.float64)
for c in range(len(sv_list)):
    abs_means += np.mean(np.abs(sv_list[c]), axis=0)
abs_means /= max(len(sv_list), 1)

# ---- 4) Plot global importance ----
topk = min(30, len(feature_names))
top_idx = np.argsort(abs_means)[::-1][:topk]
top_features = [feature_names[i] for i in top_idx]
top_scores = abs_means[top_idx]

plt.figure(figsize=(10, 8))
plt.barh(top_features[::-1], top_scores[::-1])
plt.title("SHAP Global Feature Importance (mean |SHAP| across classes)")
plt.xlabel("mean(|SHAP|)")
plt.tight_layout()
plt.savefig("shap_global_importance_FIXED.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: shap_global_importance_FIXED.png")

# ---- 5) Beeswarm for a valid class index ----
# Pick the most frequent class in your X_explain pool if possible
try:
    most_common = int(pd.Series(y_test_pool[:len(X_explain)]).value_counts().idxmax())
except Exception:
    most_common = 0

# But if SHAP returned fewer classes, clamp index
c_show = min(most_common, len(sv_list)-1)

print("Beeswarm using class index:", c_show, " | name (if exists):", class_names[c_show] if c_show < len(class_names) else "NA")

shap.summary_plot(
    sv_list[c_show],
    X_explain,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_beeswarm_FIXED.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: shap_beeswarm_FIXED.png")

