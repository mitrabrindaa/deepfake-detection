"""Train Xception (+ optional EfficientNet) with two-phase fine-tuning, MixUp, AMP, TTA eval."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .data import (
    FaceDeepFakeDataset,
    collect_samples,
    get_eval_augmentation,
    get_train_augmentation,
    mixup_criterion,
    mixup_data,
    set_seed,
    stratified_split,
)
from .model import EnsembleDetector, XceptionDeepFake, build_model
from .plots import (
    save_confusion_matrix_figure,
    save_history_json,
    save_per_class_metrics_bar,
    save_training_curves,
)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dirs():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _fusion_logits(lx: torch.Tensor, le: torch.Tensor) -> torch.Tensor:
    w = config.AUX_WEIGHT
    return (1.0 - w) * lx + w * le


def forward_batch(
    model: nn.Module,
    x: torch.Tensor,
    ensemble: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if ensemble:
        assert isinstance(model, EnsembleDetector)
        return model.xcep(x), model.eff(x)
    return model(x)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_mixup: bool,
    mixup_alpha: float,
    ensemble: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    use_amp = scaler is not None and device.type == "cuda"

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_a, y_b, lam = y, y, 1.0
        if use_mixup and np.random.rand() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                if ensemble:
                    lx, le = forward_batch(model, x, True)
                    if lam < 1.0:
                        loss = 0.5 * (
                            mixup_criterion(criterion, lx, y_a, y_b, lam)
                            + mixup_criterion(criterion, le, y_a, y_b, lam)
                        )
                    else:
                        loss = 0.5 * (criterion(lx, y) + criterion(le, y))
                else:
                    logits = forward_batch(model, x, False)
                    if lam < 1.0:
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    else:
                        loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if ensemble:
                lx, le = forward_batch(model, x, True)
                if lam < 1.0:
                    loss = 0.5 * (
                        mixup_criterion(criterion, lx, y_a, y_b, lam)
                        + mixup_criterion(criterion, le, y_a, y_b, lam)
                    )
                else:
                    loss = 0.5 * (criterion(lx, y) + criterion(le, y))
            else:
                logits = forward_batch(model, x, False)
                if lam < 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ensemble: bool,
    use_tta: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    all_prob: list[float] = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y_np = y.numpy()

        def _probs(xb: torch.Tensor) -> torch.Tensor:
            if ensemble:
                lx, le = model.xcep(xb), model.eff(xb)
                logits = _fusion_logits(lx, le)
            else:
                logits = model(xb)
            return torch.softmax(logits, dim=1)[:, 1]

        p = _probs(x)
        if use_tta:
            xf = torch.flip(x, dims=[3])
            p = 0.5 * (p + _probs(xf))

        pred = (p.cpu().numpy() >= 0.5).astype(np.int64)
        all_pred.extend(pred.tolist())
        all_true.extend(y_np.tolist())
        all_prob.extend(p.cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    return (
        acc,
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob),
    )


def _build_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_backbone: float,
    ensemble: bool,
    phase2: bool,
) -> torch.optim.Optimizer:
    if not phase2:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head,
            weight_decay=config.WEIGHT_DECAY,
        )

    head_params = []
    back_params = []
    if ensemble:
        assert isinstance(model, EnsembleDetector)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "xcep.backbone.output" in name or "eff.backbone.classifier" in name:
                head_params.append(p)
            else:
                back_params.append(p)
    else:
        assert isinstance(model, XceptionDeepFake)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone.output"):
                head_params.append(p)
            else:
                back_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": back_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head * 0.5},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def run_training(args: argparse.Namespace) -> None:
    set_seed(config.SEED)
    _ensure_dirs()
    device = _device()
    if device.type == "cpu":
        n = os.cpu_count() or 4
        torch.set_num_threads(max(1, n))
        torch.set_num_interop_threads(max(1, min(4, n // 2)))

    paths, labels = collect_samples(Path(args.data_dir))
    if len(paths) < 20:
        raise SystemExit("Not enough images under dataset/real and dataset/fake.")

    max_pc: int | None = None
    if args.quick:
        max_pc = 200
    elif args.max_per_class is not None:
        max_pc = args.max_per_class
    if max_pc is not None:
        rng = np.random.default_rng(config.SEED)
        paths_np = np.array(paths)
        labels_np = np.array(labels)
        keep: list[int] = []
        for c in (0, 1):
            cls_idx = np.where(labels_np == c)[0]
            take = min(max_pc, len(cls_idx))
            keep.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(paths, labels, config.SEED)

    split_payload = {
        "train": [str(p) for p in X_train],
        "val": [str(p) for p in X_val],
        "test": [str(p) for p in X_test],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    with open(config.ARTIFACTS_DIR / "split.json", "w", encoding="utf-8") as f:
        json.dump({**split_payload, "meta": {"n_total": len(paths)}}, f, indent=0)

    train_tf = get_train_augmentation(config.IMG_SIZE)
    eval_tf = get_eval_augmentation(config.IMG_SIZE)

    train_ds = FaceDeepFakeDataset(X_train, y_train, train_tf, True)
    val_ds = FaceDeepFakeDataset(X_val, y_val, eval_tf, False)
    test_ds = FaceDeepFakeDataset(X_test, y_test, eval_tf, False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    ensemble = args.ensemble and config.USE_ENSEMBLE_AUX
    model = build_model(use_ensemble=ensemble, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val = 0.0
    best_path = config.CHECKPOINT_DIR / "best.pt"
    resume_path = config.CHECKPOINT_DIR / "resume.pt"
    patience = 0

    history: dict = {
        "phase": [],
        "train_loss": [],
        "val_acc": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
    }

    epochs_p1 = args.epochs_p1
    epochs_p2 = args.epochs_p2
    p1_epochs_done = 0
    p2_epochs_done = 0
    training_phase = "p1"

    resume_data = None
    if args.resume is not None:
        rpath = Path(args.resume)
        if not rpath.is_file():
            raise SystemExit(f"--resume file not found: {rpath}")
        resume_data = torch.load(rpath, map_location=device, weights_only=False)
        if bool(resume_data.get("ensemble")) != bool(ensemble):
            raise SystemExit(
                "Resume checkpoint ensemble setting does not match this run "
                "(use the same flags as before, e.g. same --no-ensemble)."
            )
        model.load_state_dict(resume_data["model"])
        for k in history:
            history[k] = list(resume_data["history"][k])
        best_val = float(resume_data["best_val"])
        patience = int(resume_data["patience"])
        p1_epochs_done = int(resume_data["p1_epochs_done"])
        p2_epochs_done = int(resume_data["p2_epochs_done"])
        training_phase = str(resume_data["training_phase"])
        if resume_data.get("epochs_p1") != epochs_p1 or resume_data.get("epochs_p2") != epochs_p2:
            print(
                "Warning: --epochs-p1 / --epochs-p2 differ from the saved resume; "
                "scheduler length may not match the original run."
            )
        if scaler is not None and resume_data.get("scaler") is not None:
            scaler.load_state_dict(resume_data["scaler"])
        tp = training_phase
        if tp not in ("p1", "p2"):
            raise SystemExit(f"Invalid training_phase in resume: {tp}")
        if tp == "p2" and p1_epochs_done < epochs_p1:
            raise SystemExit("Invalid resume: phase p2 but p1_epochs_done < epochs_p1.")
        print(
            f"Resuming: phase={training_phase}, p1_epochs_done={p1_epochs_done}/{epochs_p1}, "
            f"p2_epochs_done={p2_epochs_done}/{epochs_p2}, best_val={best_val:.4f}"
        )

    def _save_resume(
        training_phase_now: str,
        p1d: int,
        p2d: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "training_phase": training_phase_now,
            "p1_epochs_done": p1d,
            "p2_epochs_done": p2d,
            "epochs_p1": epochs_p1,
            "epochs_p2": epochs_p2,
            "best_val": best_val,
            "patience": patience,
            "history": {k: list(v) for k, v in history.items()},
            "ensemble": ensemble,
        }
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        torch.save(payload, resume_path)

    def _log_val_epoch(phase_id: int, tr_loss: float) -> None:
        val_acc, vy, vpred, _ = evaluate(model, val_loader, device, ensemble, use_tta=True)
        pm = float(precision_score(vy, vpred, average="macro", zero_division=0))
        rm = float(recall_score(vy, vpred, average="macro", zero_division=0))
        fm = float(f1_score(vy, vpred, average="macro", zero_division=0))
        history["phase"].append(phase_id)
        history["train_loss"].append(float(tr_loss))
        history["val_acc"].append(float(val_acc))
        history["val_precision_macro"].append(pm)
        history["val_recall_macro"].append(rm)
        history["val_f1_macro"].append(fm)

    # Phase 1 — frozen backbones
    if p1_epochs_done < epochs_p1:
        if ensemble:
            model.freeze_backbones()
        else:
            model.freeze_backbone()

        optimizer = _build_optimizer(model, config.LR_HEAD, config.LR_BACKBONE, ensemble, phase2=False)
        if resume_data is not None and training_phase == "p1" and resume_data.get("optimizer"):
            optimizer.load_state_dict(resume_data["optimizer"])

        for epoch in range(p1_epochs_done, epochs_p1):
            tr_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                args.mixup and config.USE_MIXUP,
                config.MIXUP_ALPHA,
                ensemble,
            )
            _log_val_epoch(1, tr_loss)
            val_acc = history["val_acc"][-1]
            print(f"Phase1 [{epoch+1}/{epochs_p1}] train_loss={tr_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "phase": 1,
                        "ensemble": ensemble,
                    },
                    best_path,
                )
                patience = 0
            else:
                patience += 1
            p1_epochs_done = epoch + 1
            _save_resume("p1", p1_epochs_done, 0, optimizer, None)
        p1_epochs_done = epochs_p1

    # Phase 2 — load best phase-1 weights when starting fine-tuning (not when resuming mid–phase 2)
    if p2_epochs_done == 0 and best_path.exists():
        ckpt_best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_best["model"])

    # Phase 2 — fine-tune last blocks + heads (early layers frozen for stability)
    if ensemble:
        assert isinstance(model, EnsembleDetector)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.xcep.backbone.output.parameters():
            p.requires_grad = True
        for p in model.eff.backbone.classifier.parameters():
            p.requires_grad = True
        model.xcep.unfreeze_last_stages()
        model.eff.unfreeze_last_blocks()
    else:
        assert isinstance(model, XceptionDeepFake)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.backbone.output.parameters():
            p.requires_grad = True
        model.unfreeze_last_stages()

    optimizer = _build_optimizer(model, config.LR_HEAD, config.LR_BACKBONE, ensemble, phase2=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs_p2), eta_min=config.LR_BACKBONE * 0.1
    )
    if resume_data is not None and training_phase == "p2":
        if resume_data.get("optimizer"):
            optimizer.load_state_dict(resume_data["optimizer"])
        if resume_data.get("scheduler") is not None:
            scheduler.load_state_dict(resume_data["scheduler"])

    for epoch in range(p2_epochs_done, epochs_p2):
        tr_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.mixup and config.USE_MIXUP,
            config.MIXUP_ALPHA,
            ensemble,
        )
        scheduler.step()
        _log_val_epoch(2, tr_loss)
        val_acc = history["val_acc"][-1]
        print(
            f"Phase2 [{epoch+1}/{epochs_p2}] train_loss={tr_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "phase": 2,
                    "val_acc": val_acc,
                    "ensemble": ensemble,
                },
                best_path,
            )
            patience = 0
        else:
            patience += 1
        p2_epochs_done = epoch + 1
        _save_resume("p2", epochs_p1, p2_epochs_done, optimizer, scheduler)
        if patience >= config.EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    _, y_val_true, y_val_pred, _ = evaluate(model, val_loader, device, ensemble, use_tta=True)
    save_confusion_matrix_figure(
        y_val_true,
        y_val_pred,
        config.ARTIFACTS_DIR,
        "confusion_matrix_val",
        "Confusion matrix — validation (best checkpoint)",
    )
    save_per_class_metrics_bar(
        y_val_true,
        y_val_pred,
        config.ARTIFACTS_DIR,
        "metrics_per_class_val",
        "Precision / Recall / F1 — validation (best checkpoint)",
    )

    test_acc, yt, y_pred, y_prob = evaluate(model, test_loader, device, ensemble, use_tta=True)
    try:
        auc = roc_auc_score(yt, y_prob) if len(np.unique(yt)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    report = classification_report(yt, y_pred, target_names=["real", "fake"], digits=4)
    metrics = {
        "test_accuracy": float(test_acc),
        "test_roc_auc_fake": float(auc),
        "best_val_accuracy": float(best_val),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "ensemble": ensemble,
        "device": str(device),
    }
    print(report)
    print(json.dumps(metrics, indent=2))

    with open(config.ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "classification_report": report}, f, indent=2)

    save_training_curves(history, config.ARTIFACTS_DIR)
    save_history_json(history, config.ARTIFACTS_DIR)
    save_confusion_matrix_figure(
        yt,
        y_pred,
        config.ARTIFACTS_DIR,
        "confusion_matrix_test",
        "Confusion matrix — test set",
    )
    save_per_class_metrics_bar(
        yt,
        y_pred,
        config.ARTIFACTS_DIR,
        "metrics_per_class_test",
        "Precision / Recall / F1 — test set (macro curves on validation in training_curves.png)",
    )

    torch.save(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "ensemble": ensemble,
        },
        config.CHECKPOINT_DIR / "final.pt",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Xception deepfake detector")
    p.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs-p1", type=int, default=config.EPOCHS_PHASE1)
    p.add_argument("--epochs-p2", type=int, default=config.EPOCHS_PHASE2)
    p.add_argument("--quick", action="store_true", help="Small subset + fewer epochs for smoke test")
    p.add_argument("--no-ensemble", action="store_true", help="Xception only (no EfficientNet branch)")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Stratified cap per class (CPU-friendly; omit to use full dataset)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoints/resume.pt from an interrupted run (same --no-ensemble etc. as before).",
    )
    args = p.parse_args(argv)

    if args.quick:
        args.epochs_p1 = min(3, args.epochs_p1)
        args.epochs_p2 = min(4, args.epochs_p2)

    args.ensemble = not args.no_ensemble
    args.mixup = not args.no_mixup

    t0 = time.time()
    run_training(args)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
