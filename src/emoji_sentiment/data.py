from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random
import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

EXTERNAL_DATASETS = {
    "real_holdout": {
        "path": "datasets/real_holdout.csv",
        "text_column": "text",
        "label_column": "label",
    },
    "real_holdout_stress_600": {
        "path": "datasets/real_holdout_stress_600.csv",
        "text_column": "stress_text",
        "label_column": "label",
    },
    "emoji_sentiment": {
        "path": "emoji_sentiment_dataset.csv",
        "text_column": "sentence",
        "label_column": "sentiment",
    },
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]",
    flags=re.UNICODE,
)
EMOJI_JOINERS = re.compile(r"[\u200d\ufe0f]")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def _clean_emoji_markup(text: str) -> str:
    return EMOJI_JOINERS.sub("", str(text))


def extract_emoji_tokens(text: str) -> list[str]:
    normalized = _clean_emoji_markup(text)
    return EMOJI_PATTERN.findall(normalized)


def extract_text_only(text: str) -> str:
    normalized = _clean_emoji_markup(text)
    without_emojis = EMOJI_PATTERN.sub(" ", normalized)
    without_emojis = re.sub(r"\s+", " ", without_emojis).strip()
    return without_emojis or "[empty]"


def normalize_template(text: str) -> str:
    text_only = extract_text_only(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", text_only)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "[empty]"


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_phrase_key(text: str, drop_tokens: int = 2) -> str:
    tokens = tokenize_text(text)
    if len(tokens) <= drop_tokens:
        return " ".join(tokens) or "[empty]"
    return " ".join(tokens[drop_tokens:])


def load_training_dataframe(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="skip",
    )
    df = df.rename(columns={"comment": "text", "sentiment": "label"})
    df["label"] = df["label"].astype(str).str.strip().str.title().map(LABEL_TO_ID)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["text", "label"]).copy()

    df["raw_text"] = df["text"].astype(str)
    df["text_only"] = df["raw_text"].map(extract_text_only)
    df["emoji_tokens"] = df["raw_text"].map(extract_emoji_tokens)
    df["template_key"] = df["raw_text"].map(normalize_template)
    df["phrase_key"] = df["text_only"].map(build_phrase_key)
    return df.reset_index(drop=True)


def load_external_dataframe(
    path: str | Path,
    text_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    df = raw_df.rename(columns={text_column: "text", label_column: "label"}).copy()

    input_rows = len(df)
    df["label_raw"] = df["label"].astype(str).str.strip().str.title()
    df["label"] = df["label_raw"].map(LABEL_TO_ID)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["raw_text"] = df["text"].astype(str)
    df["text_only"] = df["raw_text"].map(extract_text_only)
    df["emoji_tokens"] = df["raw_text"].map(extract_emoji_tokens)
    df["template_key"] = df["raw_text"].map(normalize_template)
    df["phrase_key"] = df["text_only"].map(build_phrase_key)

    stats = {
        "input_rows": input_rows,
        "kept_rows": len(df),
        "dropped_rows": input_rows - len(df),
    }
    return df.reset_index(drop=True), stats


def split_dataframe(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
    strategy: str = "grouped",
    group_column: str = "phrase_key",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if strategy == "random":
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        val_ratio = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df["label"],
        )
    elif strategy == "grouped":
        groups = (
            df.groupby(group_column, as_index=False)
            .agg(label=("label", "first"), row_count=("label", "size"))
            .reset_index(drop=True)
        )
        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=random_state,
            stratify=groups["label"],
        )
        val_ratio = val_size / (train_size + val_size)
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_groups["label"],
        )

        train_keys = set(train_groups[group_column])
        val_keys = set(val_groups[group_column])
        test_keys = set(test_groups[group_column])

        train_df = df[df[group_column].isin(train_keys)].copy()
        val_df = df[df[group_column].isin(val_keys)].copy()
        test_df = df[df[group_column].isin(test_keys)].copy()
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    split_stats: dict[str, int | str] = {
        "strategy": strategy,
        "group_column": group_column,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_templates": int(train_df[group_column].nunique()),
        "val_templates": int(val_df[group_column].nunique()),
        "test_templates": int(test_df[group_column].nunique()),
        "template_overlap_train_val": int(
            len(set(train_df[group_column]) & set(val_df[group_column]))
        ),
        "template_overlap_train_test": int(
            len(set(train_df[group_column]) & set(test_df[group_column]))
        ),
        "template_overlap_val_test": int(
            len(set(val_df[group_column]) & set(test_df[group_column]))
        ),
    }

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_stats,
    )


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"

    @classmethod
    def build(
        cls,
        sequences: list[list[str]],
        *,
        min_freq: int,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ) -> "Vocabulary":
        counts: dict[str, int] = {}
        for tokens in sequences:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        token_to_id = {pad_token: 0, unk_token: 1}
        for token in sorted(token for token, count in counts.items() if count >= min_freq):
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, pad_token=pad_token, unk_token=unk_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, tokens: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        trimmed = tokens[:max_length]
        ids = [self.token_to_id.get(token, self.unk_id) for token in trimmed]
        length = max(1, len(ids))
        padded = ids + [self.pad_id] * (max_length - len(ids))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.token_to_id)


class TextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_length: int,
        token_dropout: float = 0.0,
    ):
        self.samples = df["text_only"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_length = max_length
        self.token_dropout = token_dropout

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = tokenize_text(self.samples[idx])
        if self.token_dropout > 0 and tokens:
            dropped = [
                token if random.random() > self.token_dropout else self.vocab.unk_token
                for token in tokens
            ]
            if any(token != self.vocab.unk_token for token in dropped):
                tokens = dropped
        input_ids, length = self.vocab.encode(tokens, self.max_length)
        return {
            "input_ids": input_ids,
            "lengths": length,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EmojiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_length: int,
        token_dropout: float = 0.0,
    ):
        self.samples = df["emoji_tokens"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_length = max_length
        self.token_dropout = token_dropout

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[idx] or ["[NO_EMOJI]"]
        if self.token_dropout > 0 and tokens != ["[NO_EMOJI]"]:
            dropped = [
                token if random.random() > self.token_dropout else self.vocab.unk_token
                for token in tokens
            ]
            if any(token != self.vocab.unk_token for token in dropped):
                tokens = dropped
        emoji_ids, emoji_length = self.vocab.encode(tokens, self.max_length)
        return {
            "emoji_ids": emoji_ids,
            "emoji_lengths": emoji_length,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CombinedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_vocab: Vocabulary,
        emoji_vocab: Vocabulary,
        text_max_length: int,
        emoji_max_length: int,
        text_token_dropout: float = 0.0,
        emoji_token_dropout: float = 0.0,
    ):
        self.text_dataset = TextDataset(
            df,
            text_vocab,
            text_max_length,
            token_dropout=text_token_dropout,
        )
        self.emoji_dataset = EmojiDataset(
            df,
            emoji_vocab,
            emoji_max_length,
            token_dropout=emoji_token_dropout,
        )
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text_item = self.text_dataset[idx]
        emoji_item = self.emoji_dataset[idx]
        return {
            "input_ids": text_item["input_ids"],
            "lengths": text_item["lengths"],
            "emoji_ids": emoji_item["emoji_ids"],
            "emoji_lengths": emoji_item["emoji_lengths"],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@dataclass
class LoaderBundle:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    test: DataLoader
    text_vocab: Vocabulary
    emoji_vocab: Vocabulary


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    text_max_length: int,
    emoji_max_length: int,
    min_text_freq: int,
    min_emoji_freq: int,
    pin_memory: bool,
    text_token_dropout: float = 0.0,
    emoji_token_dropout: float = 0.0,
) -> dict[str, LoaderBundle]:
    train_text_tokens = [tokenize_text(text) for text in train_df["text_only"].astype(str)]
    train_emoji_tokens = [tokens or ["[NO_EMOJI]"] for tokens in train_df["emoji_tokens"]]

    text_vocab = Vocabulary.build(train_text_tokens, min_freq=min_text_freq)
    emoji_vocab = Vocabulary.build(
        train_emoji_tokens,
        min_freq=min_emoji_freq,
    )
    if "[NO_EMOJI]" not in emoji_vocab.token_to_id:
        emoji_vocab.token_to_id["[NO_EMOJI]"] = len(emoji_vocab.token_to_id)

    loader_factory: dict[str, Callable[[pd.DataFrame], Dataset]] = {
        "text": lambda frame: TextDataset(frame, text_vocab, text_max_length),
        "emoji": lambda frame: EmojiDataset(frame, emoji_vocab, emoji_max_length),
        "combined": lambda frame: CombinedDataset(
            frame,
            text_vocab,
            emoji_vocab,
            text_max_length,
            emoji_max_length,
        ),
    }

    bundles: dict[str, LoaderBundle] = {}
    for name, dataset_builder in loader_factory.items():
        if name == "text":
            train_dataset = TextDataset(
                train_df,
                text_vocab,
                text_max_length,
                token_dropout=text_token_dropout,
            )
        elif name == "emoji":
            train_dataset = EmojiDataset(
                train_df,
                emoji_vocab,
                emoji_max_length,
                token_dropout=emoji_token_dropout,
            )
        else:
            train_dataset = CombinedDataset(
                train_df,
                text_vocab,
                emoji_vocab,
                text_max_length,
                emoji_max_length,
                text_token_dropout=text_token_dropout,
                emoji_token_dropout=emoji_token_dropout,
            )
        train_eval_dataset = dataset_builder(train_df)
        val_dataset = dataset_builder(val_df)
        test_dataset = dataset_builder(test_df)

        bundles[name] = LoaderBundle(
            train=_make_loader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            train_eval=_make_loader(
                train_eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            val=_make_loader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            test=_make_loader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            text_vocab=text_vocab,
            emoji_vocab=emoji_vocab,
        )

    return bundles
