"""Training loop.

Run from the notebook as:
    from src.train import train
    train(cfg)

Or from the command line:
    python -m src.train --config configs/config.yaml

Design notes:
- Loss: plain CrossEntropyLoss. The benign/malignant split is ~55/45, a mild
  imbalance where class weighting buys little recall but harms probability
  calibration (and we threshold probabilities for sensitivity/specificity).
  Weighting is available behind cfg.train.use_class_weights for A/B testing.
- Resumable: every epoch writes last_checkpoint.pt (model + optimizer +
  scheduler + scaler + epoch + best_auc) to Drive, so a Colab disconnect
  costs at most the in-progress epoch. best_model.pt tracks the best val AUC.
- Logging: pass log_csv=<path> to append one row per epoch (epoch, losses,
  AUCs, learning rate, epoch seconds). The rows are also kept in the
  checkpoint under "history", so a resumed run rewrites the file exactly
  rather than leaving a gap or a duplicate epoch.
- run_tag identifies the dataset/run a checkpoint belongs to (e.g. "v2").
  A checkpoint whose tag differs is NOT resumed from: it would silently
  continue a different run on different data.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.amp import GradScaler, autocast

from src.config import ensure_dirs, load_config
from src.data import make_loader
from src.model import build_model

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; degrade to a no-op wrapper.
    def tqdm(iterable, **kwargs):
        return iterable


def _make_optimizer(cfg, model):
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)


def _make_criterion(cfg, train_loader, device):
    """Plain CE by default; inverse-frequency weighted CE if explicitly enabled."""
    if getattr(cfg.train, "use_class_weights", False):
        labels = train_loader.dataset.df[cfg.data.label_column].to_numpy()
        counts = np.bincount(labels, minlength=cfg.model.num_classes)
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Using weighted CE: class weights = {weights.round(3).tolist()}")
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


def _auc(y_true, y_prob):
    """ROC-AUC guarded against a single-class batch/epoch."""
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                     use_amp, pos, desc=None):
    """One training pass. Returns (mean_loss, auc) computed from the
    on-the-fly predictions made during the epoch (approximate, since weights
    update as we go — standard for a cheap train-AUC readout).

    The tqdm bar uses leave=False so it erases itself after the epoch, leaving
    only the one-line summary printed by the caller.
    """
    model.train()
    running_loss = 0.0
    probs_all, labels_all = [], []
    amp_dev = device.type

    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(amp_dev, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        probs_all.append(torch.softmax(outputs.detach().float(), dim=1)[:, pos].cpu())
        labels_all.append(labels.cpu())

    y_prob = torch.cat(probs_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    return running_loss / len(loader.dataset), _auc((y_true == pos).astype(int), y_prob)


@torch.no_grad()
def _validate(model, loader, criterion, device, use_amp, pos):
    """One validation pass. Returns (mean_loss, auc, accuracy)."""
    model.eval()
    running_loss = 0.0
    probs_all, preds_all, labels_all = [], [], []
    amp_dev = device.type

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(amp_dev, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs_all.append(torch.softmax(outputs.float(), dim=1)[:, pos].cpu())
        preds_all.append(outputs.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())

    y_prob = torch.cat(probs_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    val_loss = running_loss / len(loader.dataset)
    return val_loss, _auc((y_true == pos).astype(int), y_prob), accuracy_score(y_true, y_pred)


LOG_FIELDS = ["run_tag", "epoch", "train_loss", "train_auc", "val_loss",
              "val_auc", "lr", "epoch_seconds"]


def _write_log(log_csv, history):
    """Rewrite the whole log from `history` (one dict per completed epoch)."""
    if not log_csv:
        return
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(history)


def train(cfg, resume=True, run_tag=None, log_csv=None):
    torch.manual_seed(cfg.project.seed)
    ensure_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos = cfg.evaluate.positive_class

    model = build_model(cfg).to(device)
    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    criterion = _make_criterion(cfg, train_loader, device)
    optimizer = _make_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs)
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    best_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pt")
    last_path = os.path.join(cfg.paths.checkpoint_dir, "last_checkpoint.pt")

    start_epoch = 0
    best_auc = 0.0
    epochs_no_improve = 0
    history = []

    # --- Resume from the last checkpoint if one exists on Drive ---
    # Only resume a checkpoint from the same run: a tag mismatch means the file
    # belongs to a different dataset version, so resuming would mix the two.
    if resume and os.path.isfile(last_path):
        ckpt_tag = torch.load(last_path, map_location="cpu", weights_only=False).get("run_tag")
        if ckpt_tag != run_tag:
            print(f"Ignoring {last_path}: run_tag {ckpt_tag!r} does not match {run_tag!r}; "
                  f"training from ImageNet-pretrained weights.")
            resume = False
    if resume and os.path.isfile(last_path):
        # weights_only=False: our own checkpoint carries non-tensor state
        # (epoch, best_auc, optimizer/scheduler dicts). Trusted source.
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["best_auc"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = list(ckpt.get("history", []))
        print(f"Resumed from {last_path}: starting at epoch {start_epoch + 1}, "
              f"best_auc={best_auc:.4f}")
        _write_log(log_csv, history)   # drop any rows past the checkpoint

    if start_epoch >= cfg.train.epochs:
        print("Already trained for the configured number of epochs.")
        return model
    if log_csv and not history:
        _write_log(log_csv, [])        # fresh run: start the log with a header only

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        lr_epoch = optimizer.param_groups[0]["lr"]   # the rate used DURING this epoch
        train_loss, train_auc = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, pos,
            desc=f"Epoch {epoch + 1:02d}/{cfg.train.epochs} [train]")
        scheduler.step()
        val_loss, val_auc, val_acc = _validate(
            model, val_loader, criterion, device, use_amp, pos)

        improved = val_auc > best_auc
        if improved:
            best_auc = float(val_auc)  # plain float so checkpoints carry no numpy scalars
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # One clean line per epoch (reads well in a screen recording).
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{cfg.train.epochs} | "
            f"train loss {train_loss:.4f} auc {train_auc:.4f} | "
            f"val loss {val_loss:.4f} auc {val_auc:.4f} acc {val_acc:.4f} | "
            f"lr {lr:.2e} | {time.time() - t0:4.0f}s"
            + ("  *best" if improved else "")
        )

        history.append({"run_tag": run_tag, "epoch": epoch + 1,
                        "train_loss": round(train_loss, 6), "train_auc": round(float(train_auc), 6),
                        "val_loss": round(val_loss, 6), "val_auc": round(float(val_auc), 6),
                        "lr": lr_epoch, "epoch_seconds": round(time.time() - t0, 1)})

        # Always save 'last' (for resume); save 'best' only on improvement.
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_auc": best_auc,
            "epochs_no_improve": epochs_no_improve,
            "run_tag": run_tag,
            "history": history,
        }
        torch.save(state, last_path)
        _write_log(log_csv, history)
        if improved:
            torch.save(state, best_path)

        if epochs_no_improve >= cfg.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(no val AUC improvement for {epochs_no_improve} epochs).")
            break

    print(f"Done. Best val AUC = {best_auc:.4f} -> {best_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()
