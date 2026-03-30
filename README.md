# DeepFake Detection (Xception + optional EfficientNet ensemble)

Binary classifier for face-crop images (`real` vs `fake`) with two-phase fine-tuning and simple inference.

## Project layout

- `src/train.py`: training entrypoint (`python -m src.train`)
- `src/config.py`: hyperparameters + paths
- `preprocess.py`: validate your dataset images (optionally delete corrupt files)
- `inference.py`: run inference on image(s) or a folder
- `dataset/`: your data (ignored by `.gitignore`)
- `artifacts/`: metrics/plots/logs (ignored by `.gitignore`)
- `checkpoints/`: saved models (ignored by `.gitignore`)

## Dataset format

Put **face-crop images** here:

```text
dataset/
  real/
    *.jpg|*.png|*.jpeg
  fake/
    *.jpg|*.png|*.jpeg
```

## Setup

Create a venv (recommended), then install deps:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## (Optional) validate dataset files

```bash
python preprocess.py
```

If you want it to delete corrupt/unreadable files:

```bash
python preprocess.py --remove-bad
```

## Train

### Fast demo (mentor deadline friendly)

- **150 per class cap**
- **short epochs**
- writes a timestamped log (won’t overwrite old logs)

```powershell
$log="artifacts\training_full_run_fastdemo_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"; `
python -m src.train --max-per-class 150 --epochs-p1 3 --epochs-p2 4 2>&1 | Tee-Object -FilePath $log
```

### Full run (use more data / epochs)

By default, epochs come from `src/config.py` (`EPOCHS_PHASE1`, `EPOCHS_PHASE2`). To train on the full dataset, omit `--max-per-class`.

```powershell
$log="artifacts\training_full_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"; `
python -m src.train 2>&1 | Tee-Object -FilePath $log
```

### Notes

- **Ensemble on/off**:
  - default: ensemble enabled (Xception + EfficientNet)
  - Xception-only: add `--no-ensemble`
- **MixUp on/off**:
  - default: enabled
  - disable: add `--no-mixup`
- `--quick` is a smoke-test flag (it uses a built-in subset and forces smaller epoch counts).

## Outputs

Training creates:

- `checkpoints/best.pt`, `checkpoints/resume.pt`, `checkpoints/final.pt`
- `artifacts/metrics.json` (includes classification report)
- `artifacts/history.json`
- `artifacts/split.json`
- `artifacts/figures/*.png` (training curves + confusion matrices)

## Inference

Run on one image:

```bash
python inference.py path\to\image.jpg
```

Run on a folder:

```bash
python inference.py path\to\folder
```

Use a specific checkpoint:

```bash
python inference.py checkpoints\best.pt path\to\image_or_folder
```

The script prints: `fake|real` plus `P(fake)=...` (threshold is 0.5).

