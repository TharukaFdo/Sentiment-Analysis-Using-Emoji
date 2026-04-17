from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn import functional as F
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _forward_pipeline(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    pipeline: str,
) -> torch.Tensor:
    if pipeline == "text":
        return model(batch["input_ids"], batch["lengths"])
    if pipeline == "emoji":
        return model(batch["emoji_ids"], batch["emoji_lengths"])
    return model(
        batch["input_ids"],
        batch["lengths"],
        batch["emoji_ids"],
        batch["emoji_lengths"],
    )


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_probs_a = F.log_softmax(logits_a, dim=1)
    log_probs_b = F.log_softmax(logits_b, dim=1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    kl_ab = F.kl_div(log_probs_a, probs_b, reduction="batchmean")
    kl_ba = F.kl_div(log_probs_b, probs_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def _corrupt_emoji_batch(
    emoji_ids: torch.Tensor,
    emoji_lengths: torch.Tensor,
    *,
    vocab_size: int,
    pad_id: int,
    no_emoji_id: int,
    removal_prob: float,
    swap_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrupted_ids = emoji_ids.clone()
    corrupted_lengths = emoji_lengths.clone()
    if removal_prob <= 0.0 and swap_prob <= 0.0:
        return corrupted_ids, corrupted_lengths

    valid_ids = torch.arange(vocab_size, device=emoji_ids.device)
    valid_ids = valid_ids[(valid_ids > 1) & (valid_ids != no_emoji_id) & (valid_ids != pad_id)]
    if valid_ids.numel() == 0:
        valid_ids = torch.tensor([no_emoji_id], device=emoji_ids.device, dtype=torch.long)

    batch_size, max_length = emoji_ids.shape
    for row_index in range(batch_size):
        active_length = max(1, int(emoji_lengths[row_index].item()))
        draw = torch.rand((), device=emoji_ids.device).item()
        if draw < removal_prob:
            corrupted_ids[row_index].fill_(pad_id)
            corrupted_ids[row_index, 0] = no_emoji_id
            corrupted_lengths[row_index] = 1
            continue
        if draw < removal_prob + swap_prob:
            random_positions = torch.randint(
                valid_ids.numel(),
                (active_length,),
                device=emoji_ids.device,
            )
            corrupted_ids[row_index, :active_length] = valid_ids[random_positions]
            if active_length < max_length:
                corrupted_ids[row_index, active_length:] = pad_id

    return corrupted_ids, corrupted_lengths


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    pipeline: str,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
    combined_corruption_loss_weight: float = 0.0,
    combined_invariance_weight: float = 0.0,
    combined_emoji_removal_prob: float = 0.0,
    combined_emoji_swap_prob: float = 0.0,
    combined_emoji_vocab_size: int = 0,
    combined_emoji_pad_id: int = 0,
    combined_emoji_no_emoji_id: int = 0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = _forward_pipeline(model, batch, pipeline)
            clean_loss = criterion(logits, labels)
            loss = clean_loss
            if pipeline == "combined" and (
                combined_corruption_loss_weight > 0.0 or combined_invariance_weight > 0.0
            ):
                corrupted_emoji_ids, corrupted_emoji_lengths = _corrupt_emoji_batch(
                    batch["emoji_ids"],
                    batch["emoji_lengths"],
                    vocab_size=combined_emoji_vocab_size,
                    pad_id=combined_emoji_pad_id,
                    no_emoji_id=combined_emoji_no_emoji_id,
                    removal_prob=combined_emoji_removal_prob,
                    swap_prob=combined_emoji_swap_prob,
                )
                corrupted_batch = dict(batch)
                corrupted_batch["emoji_ids"] = corrupted_emoji_ids
                corrupted_batch["emoji_lengths"] = corrupted_emoji_lengths
                corrupted_logits = _forward_pipeline(model, corrupted_batch, pipeline)
                corrupted_loss = criterion(corrupted_logits, labels)
                consistency_loss = _symmetric_kl(logits, corrupted_logits)
                loss = (
                    clean_loss
                    + (combined_corruption_loss_weight * corrupted_loss)
                    + (combined_invariance_weight * consistency_loss)
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return total_loss / max(1, len(loader)), accuracy_score(all_labels, all_preds)


def evaluate_pipeline(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    pipeline: str,
    collect_predictions: bool = True,
) -> dict[str, float | list[int] | dict[str, object]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch["label"]
            logits = _forward_pipeline(model, batch, pipeline)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    metrics: dict[str, float | list[int] | dict[str, object]] = {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            target_names=["Negative", "Neutral", "Positive"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if collect_predictions:
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels

    return metrics


def run_training(
    *,
    models: dict[str, nn.Module],
    loaders,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device,
    output_dir: Path,
    combined_corruption_loss_weight: float = 0.0,
    combined_invariance_weight: float = 0.0,
    combined_emoji_removal_prob: float = 0.0,
    combined_emoji_swap_prob: float = 0.0,
) -> dict[str, list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    amp_enabled = device.type == "cuda"
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for pipeline, model in models.items():
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
        best_val_f1 = -1.0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_optim_loss, train_acc = train_one_epoch(
                model,
                loaders[pipeline].train,
                optimizer,
                criterion,
                device=device,
                pipeline=pipeline,
                scaler=scaler,
                amp_enabled=amp_enabled,
                combined_corruption_loss_weight=combined_corruption_loss_weight,
                combined_invariance_weight=combined_invariance_weight,
                combined_emoji_removal_prob=combined_emoji_removal_prob,
                combined_emoji_swap_prob=combined_emoji_swap_prob,
                combined_emoji_vocab_size=len(loaders[pipeline].emoji_vocab)
                if pipeline == "combined"
                else 0,
                combined_emoji_pad_id=loaders[pipeline].emoji_vocab.pad_id
                if pipeline == "combined"
                else 0,
                combined_emoji_no_emoji_id=loaders[pipeline].emoji_vocab.token_to_id.get("[NO_EMOJI]", 0)
                if pipeline == "combined"
                else 0,
            )
            train_eval = evaluate_pipeline(
                model,
                loaders[pipeline].train_eval,
                criterion,
                device=device,
                pipeline=pipeline,
                collect_predictions=False,
            )
            val_eval = evaluate_pipeline(
                model,
                loaders[pipeline].val,
                criterion,
                device=device,
                pipeline=pipeline,
                collect_predictions=False,
            )

            epoch_result = {
                "epoch": float(epoch),
                "train_optim_loss": float(train_optim_loss),
                "train_loss": float(train_eval["loss"]),
                "train_accuracy": float(train_acc),
                "train_eval_accuracy": float(train_eval["accuracy"]),
                "train_f1": float(train_eval["f1_weighted"]),
                "val_loss": float(val_eval["loss"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "val_f1": float(val_eval["f1_weighted"]),
            }
            history.append(epoch_result)

            print(
                f"[{pipeline}] epoch {epoch}/{epochs} "
                f"train_loss={epoch_result['train_loss']:.4f} "
                f"val_loss={epoch_result['val_loss']:.4f} "
                f"train_f1={epoch_result['train_f1']:.4f} "
                f"val_f1={epoch_result['val_f1']:.4f}"
            )

            if epoch_result["val_f1"] > best_val_f1:
                best_val_f1 = epoch_result["val_f1"]
                torch.save(model.state_dict(), checkpoints_dir / f"best_{pipeline}.pt")

        histories[pipeline] = history

    return histories


def plot_loss_curves(histories: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 5))
    if len(histories) == 1:
        axes = [axes]

    fig.suptitle("Training and Validation Loss Curves", fontsize=16)
    for axis, (pipeline, rows) in zip(axes, histories.items(), strict=False):
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [row["train_loss"] for row in rows]
        val_loss = [row["val_loss"] for row in rows]
        axis.plot(epochs, train_loss, label="Train Loss (eval mode)")
        axis.plot(epochs, val_loss, label="Validation Loss")
        axis.set_title(f"{pipeline.upper()} Pipeline")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
