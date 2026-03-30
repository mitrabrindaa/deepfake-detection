"""Hyperparameters and paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Images are face crops; Xception expects 299×299 (same as FaceForensics++ / many DF papers)
IMG_SIZE = 299
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# class indices: 0 = real, 1 = fake
CLASS_REAL = 0
CLASS_FAKE = 1

BATCH_SIZE = 24
NUM_WORKERS = 0  # Windows: 0 avoids multiprocessing issues; increase on Linux with GPU

# Training
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10  # rest is test
EPOCHS_PHASE1 = 10  # frozen backbone (train classification head)
EPOCHS_PHASE2 = 40  # fine-tune last Xception stages + head
LR_HEAD = 3e-4
LR_BACKBONE = 1.2e-5  # last blocks only; keep small to preserve pretrained features
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.08
DROPOUT = 0.5
HIDDEN_DIM = 512

# MixUp (reduces overfitting on small / correlated face crops)
MIXUP_ALPHA = 0.2
USE_MIXUP = True

# Optional second backbone for late fusion (set False for Xception-only)
USE_ENSEMBLE_AUX = True
AUX_WEIGHT = 0.45  # weight on aux logits when averaging (1-AUX on primary)

EARLY_STOP_PATIENCE = 12
