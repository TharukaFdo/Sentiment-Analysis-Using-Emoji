from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emoji_sentiment import (  # noqa: E402
    EXTERNAL_DATASETS,
    build_dataloaders,
    build_models,
    evaluate_pipeline,
    load_external_dataframe,
    load_training_dataframe,
    plot_loss_curves,
    run_training,
    save_json,
    set_seed,
    split_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local emoji sentiment models.")
    parser.add_argument("--train-csv", type=Path, default=REPO_ROOT / "dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--run-name", type=str, default="grouped_bilstm")
    parser.add_argument("--split-strategy", choices=["grouped", "random"], default="grouped")
    parser.add_argument(
        "--group-column",
        choices=["phrase_key", "template_key"],
        default="phrase_key",
        help="Synthetic-data grouping key used when split-strategy=grouped.",
    )
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--text-max-length", type=int, default=24)
    parser.add_argument("--emoji-max-length", type=int, default=12)
    parser.add_argument("--min-text-freq", type=int, default=2)
    parser.add_argument("--min-emoji-freq", type=int, default=2)
    parser.add_argument("--text-token-dropout", type=float, default=0.15)
    parser.add_argument("--emoji-token-dropout", type=float, default=0.10)
    parser.add_argument("--combined-emoji-branch-dropout", type=float, default=0.35)
    parser.add_argument("--combined-emoji-scale", type=float, default=0.45)
    parser.add_argument("--combined-corruption-loss-weight", type=float, default=0.40)
    parser.add_argument("--combined-invariance-weight", type=float, default=0.20)
    parser.add_argument("--combined-emoji-removal-prob", type=float, default=0.20)
    parser.add_argument("--combined-emoji-swap-prob", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def prepare_run_dir(base_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    run_dir = prepare_run_dir(args.output_dir, args.run_name)
    print(f"Saving artifacts to: {run_dir}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = load_training_dataframe(args.train_csv)
    train_split, val_split, test_split, split_stats = split_dataframe(
        train_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        strategy=args.split_strategy,
        group_column=args.group_column,
    )

    save_json(run_dir / "config.json", vars(args))
    save_json(run_dir / "split_stats.json", split_stats)

    print(
        "Split summary:",
        split_stats,
    )

    loaders = build_dataloaders(
        train_split,
        val_split,
        test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_max_length=args.text_max_length,
        emoji_max_length=args.emoji_max_length,
        min_text_freq=args.min_text_freq,
        min_emoji_freq=args.min_emoji_freq,
        pin_memory=device.type == "cuda",
        text_token_dropout=args.text_token_dropout,
        emoji_token_dropout=args.emoji_token_dropout,
    )

    save_json(
        run_dir / "vocab_stats.json",
        {
            "text_vocab_size": len(loaders["text"].text_vocab),
            "emoji_vocab_size": len(loaders["emoji"].emoji_vocab),
        },
    )

    models = build_models(
        text_vocab_size=len(loaders["text"].text_vocab),
        text_pad_id=loaders["text"].text_vocab.pad_id,
        emoji_vocab_size=len(loaders["emoji"].emoji_vocab),
        emoji_pad_id=loaders["emoji"].emoji_vocab.pad_id,
        combined_emoji_branch_dropout=args.combined_emoji_branch_dropout,
        combined_emoji_scale=args.combined_emoji_scale,
    )
    histories = run_training(
        models=models,
        loaders=loaders,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        device=device,
        output_dir=run_dir,
        combined_corruption_loss_weight=args.combined_corruption_loss_weight,
        combined_invariance_weight=args.combined_invariance_weight,
        combined_emoji_removal_prob=args.combined_emoji_removal_prob,
        combined_emoji_swap_prob=args.combined_emoji_swap_prob,
    )
    save_json(run_dir / "training_history.json", histories)
    plot_loss_curves(histories, run_dir / "loss_curves.png")

    criterion = torch.nn.CrossEntropyLoss()
    metrics_payload: dict[str, object] = {
        "internal_test": {},
        "external": {},
    }

    for pipeline, model in models.items():
        state_path = run_dir / "checkpoints" / f"best_{pipeline}.pt"
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.to(device)
        test_metrics = evaluate_pipeline(
            model,
            loaders[pipeline].test,
            criterion,
            device=device,
            pipeline=pipeline,
            collect_predictions=False,
        )
        metrics_payload["internal_test"][pipeline] = test_metrics
        print(
            f"[internal:{pipeline}] "
            f"acc={test_metrics['accuracy']:.4f} "
            f"f1={test_metrics['f1_weighted']:.4f} "
            f"loss={test_metrics['loss']:.4f}"
        )

    for dataset_name, dataset_cfg in EXTERNAL_DATASETS.items():
        external_path = REPO_ROOT / dataset_cfg["path"]
        external_df, external_stats = load_external_dataframe(
            external_path,
            text_column=dataset_cfg["text_column"],
            label_column=dataset_cfg["label_column"],
        )
        external_loaders = build_dataloaders(
            train_split,
            external_df,
            external_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_max_length=args.text_max_length,
            emoji_max_length=args.emoji_max_length,
            min_text_freq=args.min_text_freq,
            min_emoji_freq=args.min_emoji_freq,
            pin_memory=device.type == "cuda",
            text_token_dropout=0.0,
            emoji_token_dropout=0.0,
        )
        dataset_metrics: dict[str, object] = {
            "stats": external_stats,
            "results": {},
        }
        for pipeline, model in models.items():
            state_path = run_dir / "checkpoints" / f"best_{pipeline}.pt"
            model.load_state_dict(torch.load(state_path, map_location=device))
            model.to(device)
            metrics = evaluate_pipeline(
                model,
                external_loaders[pipeline].val,
                criterion,
                device=device,
                pipeline=pipeline,
                collect_predictions=False,
            )
            dataset_metrics["results"][pipeline] = metrics
            print(
                f"[external:{dataset_name}:{pipeline}] "
                f"acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1_weighted']:.4f} "
                f"loss={metrics['loss']:.4f}"
            )
        metrics_payload["external"][dataset_name] = dataset_metrics

    save_json(run_dir / "metrics.json", metrics_payload)
    print("Run complete.")


if __name__ == "__main__":
    main()
