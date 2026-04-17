from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "social_media_sentiment_test.csv"
OUTPUT_DIR = REPO_ROOT / "datasets"
GENERATION_SEED_CSV = OUTPUT_DIR / "real_generation_seed.csv"
REAL_HOLDOUT_CSV = OUTPUT_DIR / "real_holdout.csv"
OUTPUT_CSV = OUTPUT_DIR / "social_media_sentiment_dataset_v4.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "social_media_sentiment_dataset_v4_manifest.json"

RANDOM_SEED = 42
TOTAL_ROWS = 24_000
CLASS_TARGETS = {"Positive": 8_000, "Neutral": 8_000, "Negative": 8_000}
NEGATIVE_SARCASM_TARGET = 2_500
HOLDOUT_FRACTION = 0.30


LEADS = {
    "Positive": [
        ("ngl", 67),
        ("bro", 65),
        ("idk", 59),
        ("tbh", 59),
        ("fr", 55),
        ("lol", 54),
        ("omg", 52),
        ("highkey", 38),
        ("lowkey", 34),
    ],
    "Neutral": [
        ("tbh", 48),
        ("bro", 45),
        ("lol", 44),
        ("idk", 43),
        ("omg", 41),
        ("fr", 40),
        ("ngl", 36),
        ("highkey", 25),
        ("lowkey", 24),
    ],
    "Negative": [
        ("ngl", 66),
        ("bro", 62),
        ("tbh", 61),
        ("fr", 57),
        ("lol", 55),
        ("omg", 55),
        ("idk", 52),
        ("lowkey", 41),
        ("highkey", 29),
    ],
}

DESCRIPTORS = {
    "Positive": [
        ("lit", 75),
        ("awesome", 47),
        ("perfect", 45),
        ("great", 40),
        ("solid", 39),
        ("fantastic", 38),
        ("amazing", 30),
    ],
    "Neutral": [
        ("okay", 41),
        ("fine", 38),
        ("average", 34),
        ("regular", 33),
        ("standard", 29),
        ("normal", 29),
        ("alright", 18),
    ],
    "Negative": [
        ("bad", 58),
        ("awful", 54),
        ("worst", 47),
        ("trash", 45),
        ("annoying", 41),
        ("terrible", 37),
        ("pathetic", 36),
    ],
}

POSITIVE_EMOJIS = [
    "\U0001f604",
    "\U0001f525",
    "\U0001f601",
    "\u2728",
    "\U0001f60d",
    "\u2764\ufe0f",
    "\U0001f929",
    "\U0001f973",
]
NEUTRAL_EMOJIS = [
    "\U0001fae4",
    "\U0001f610",
    "\U0001f642",
    "\U0001f937",
]
NEGATIVE_EMOJIS = [
    "\U0001f624",
    "\U0001f620",
    "\U0001f61e",
    "\U0001f612",
    "\U0001f494",
    "\U0001f62d",
    "\U0001f44e",
    "\U0001f621",
]
AMBIGUOUS_EMOJIS = [
    "\U0001f642",
    "\U0001f602",
    "\U0001f525",
    "\U0001f62d",
    "\U0001f64f",
    "\U0001f914",
    "\U0001f643",
    "\U0001f62c",
    "\U0001f480",
]
SARCASM_EMOJIS = [
    "\U0001f602",
    "\U0001f643",
    "\U0001f60f",
    "\U0001f644",
    "\U0001f612",
    "\U0001f44f",
]

SARCASM_FRAMES = [
    "great job",
    "best ever",
    "wow amazing",
    "just perfect",
    "love that for me",
    "yeah sure",
]
SARCASM_TAILS = [
    "totally helpful",
    "super helpful",
    "so helpful",
    "that helped a lot",
    "exactly what i needed",
]

INTENSIFIERS = ["so", "really", "actually", "honestly"]
SOFTENERS = ["just", "kinda", "pretty", "mostly"]
NEUTRAL_STEMS = ["okay", "fine", "average", "regular", "standard", "normal"]
SHORT_OVERLAP_PHRASES = {
    "Positive": [
        "{lead} this is not bad at all",
        "{lead} this hits different",
        "{lead} this is actually good",
        "{lead} this feels right",
        "{lead} this is kinda nice",
        "{lead} this is lowkey good",
    ],
    "Neutral": [
        "{lead} this is okay i guess",
        "{lead} this is just there",
        "{lead} it is what it is",
        "{lead} this is not bad not great",
        "{lead} this feels okay",
        "{lead} this is kinda fine",
    ],
    "Negative": [
        "{lead} this is not good at all",
        "{lead} this feels off",
        "{lead} this hits wrong",
        "{lead} this is actually bad",
        "{lead} this is kinda rough",
        "{lead} this is lowkey bad",
    ],
}
CONTRADICTION_PATTERNS = {
    "Positive": [
        "{lead} this looked rough but it is actually {descriptor}",
        "{lead} thought this would be bad but it is {descriptor}",
        "{lead} i was ready to hate this but it is {descriptor}",
    ],
    "Neutral": [
        "{lead} the emoji looks dramatic but it was just {descriptor}",
        "{lead} this looks intense but it was {descriptor}",
        "{lead} this felt louder than it really was and it was {descriptor}",
    ],
    "Negative": [
        "{lead} this looked fine but it is actually {descriptor}",
        "{lead} i wanted to like it but this is {descriptor}",
        "{lead} thought this might be good but this is {descriptor}",
    ],
}

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
SPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[a-zA-Z]+")

GENERATION_SIGNAL_REASONING = {
    "real_observation": (
        "Untouched rows from the real social-media seed set. These rows anchor the dataset in observed slang, emoji use, "
        "and sentiment patterns."
    ),
    "lexical_swap": (
        "Keeps the sentiment class fixed while swapping to another real seed descriptor from the same polarity family, "
        "reducing dependence on one exact wording."
    ),
    "orthographic_noise": (
        "Injects repeated letters and small typos because the real seed data uses noisy spellings typical of short social posts."
    ),
    "mixed_resolution": (
        "Creates harder examples where early text is mild or ambiguous but the final clause resolves the sentiment, closer to "
        "how people post quick updates online."
    ),
    "hedged_neutral": (
        "Builds low-affect neutral rows with hedges such as 'just' and 'kinda' so neutral is not merely a weak version of positive."
    ),
    "balanced_neutral": (
        "Creates explicitly mixed low-intensity sentiment such as 'not bad not great', which is realistic for neutral reactions."
    ),
    "shared_emoji_overlap": (
        "Uses emojis that commonly appear across positive, neutral, and negative online posts so the emoji alone does not reveal "
        "the label."
    ),
    "short_phrase_overlap": (
        "Reuses overlapping short social-media phrase families across classes, forcing the model to look beyond one lexical shell."
    ),
    "contradictory_emoji": (
        "Pairs sentiment-bearing text with ambiguous or opposing emojis so polarity cannot be solved by emoji choice alone."
    ),
    "sarcasm_negative": (
        "Uses positive surface phrases with dismissive emoji and negative pragmatic meaning. The dataset keeps the main label as "
        "Negative and tracks sarcasm separately."
    ),
}


@dataclass(frozen=True)
class SeedRow:
    seed_id: str
    comment: str
    original_label: str
    sentiment: str
    sarcasm_flag: int
    canonical_lead: str


class ParentSampler:
    def __init__(self, rows: list[SeedRow], rng: random.Random):
        self.rows = rows[:]
        self.rng = rng
        self.index = 0
        self.rng.shuffle(self.rows)

    def next(self) -> SeedRow:
        if self.index >= len(self.rows):
            self.rng.shuffle(self.rows)
            self.index = 0
        row = self.rows[self.index]
        self.index += 1
        return row


def canonicalize_lead(token: str) -> str:
    clean = re.sub(r"[^a-z]", "", token.lower())
    clean = re.sub(r"(.)\1{2,}", r"\1\1", clean)
    if clean.startswith("highk") or clean.startswith("hihgk"):
        return "highkey"
    if clean.startswith("lowk"):
        return "lowkey"
    if clean in {"fr", "tbh", "idk", "omg", "lol", "bro", "ngl"}:
        return clean
    return clean or "idk"


def normalize_comment(text: str) -> str:
    lowered = text.strip().lower()
    lowered = SPACE_RE.sub(" ", lowered)
    return lowered


def extract_lead(text: str) -> str:
    first = text.split()[0] if text.split() else "idk"
    return canonicalize_lead(first)


def load_seed_rows() -> list[SeedRow]:
    rows: list[SeedRow] = []
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            original_label = row["label"].strip().title()
            comment = SPACE_RE.sub(" ", row["text"].strip())
            if original_label == "Sarcastic":
                sentiment = "Negative"
                sarcasm_flag = 1
            else:
                sentiment = original_label
                sarcasm_flag = 0
            rows.append(
                SeedRow(
                    seed_id=f"seed_{idx:05d}",
                    comment=comment,
                    original_label=original_label,
                    sentiment=sentiment,
                    sarcasm_flag=sarcasm_flag,
                    canonical_lead=extract_lead(comment),
                )
            )
    return rows


def allocate_holdout_counts(seed_rows: list[SeedRow], holdout_fraction: float) -> dict[str, int]:
    label_counts = Counter(row.original_label for row in seed_rows)
    raw_targets = {
        label: label_counts[label] * holdout_fraction
        for label in label_counts
    }
    holdout_total = int(round(len(seed_rows) * holdout_fraction))
    floor_targets = {label: math.floor(target) for label, target in raw_targets.items()}
    remainder = holdout_total - sum(floor_targets.values())
    ranked = sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    )
    for label, _ in ranked[:remainder]:
        floor_targets[label] += 1
    return floor_targets


def partition_seed_rows(seed_rows: list[SeedRow], rng: random.Random) -> tuple[list[SeedRow], list[SeedRow], dict[str, dict[str, int]]]:
    holdout_counts = allocate_holdout_counts(seed_rows, HOLDOUT_FRACTION)
    by_label: dict[str, list[SeedRow]] = {}
    for row in seed_rows:
        by_label.setdefault(row.original_label, []).append(row)

    generation_rows: list[SeedRow] = []
    holdout_rows: list[SeedRow] = []
    summary: dict[str, dict[str, int]] = {}
    for label, rows in by_label.items():
        local_rows = rows[:]
        rng.shuffle(local_rows)
        holdout_size = holdout_counts[label]
        holdout_slice = local_rows[:holdout_size]
        generation_slice = local_rows[holdout_size:]
        holdout_rows.extend(holdout_slice)
        generation_rows.extend(generation_slice)
        summary[label] = {
            "generation_seed": len(generation_slice),
            "holdout": len(holdout_slice),
            "total": len(local_rows),
        }

    generation_rows.sort(key=lambda row: row.seed_id)
    holdout_rows.sort(key=lambda row: row.seed_id)
    return generation_rows, holdout_rows, summary


def weighted_choice(rng: random.Random, weighted_items: list[tuple[str, int]]) -> str:
    total = sum(weight for _, weight in weighted_items)
    pivot = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in weighted_items:
        cumulative += weight
        if cumulative >= pivot:
            return item
    return weighted_items[-1][0]


def sample_lead(sentiment: str, rng: random.Random, parent: SeedRow | None = None) -> str:
    if parent is not None and rng.random() < 0.45:
        return parent.canonical_lead
    return weighted_choice(rng, LEADS[sentiment])


def sample_descriptor(sentiment: str, rng: random.Random) -> str:
    return weighted_choice(rng, DESCRIPTORS[sentiment])


def contradictory_pool(sentiment: str) -> list[str]:
    if sentiment == "Positive":
        return NEGATIVE_EMOJIS[:4] + AMBIGUOUS_EMOJIS
    if sentiment == "Negative":
        return POSITIVE_EMOJIS[:4] + AMBIGUOUS_EMOJIS
    if sentiment == "Neutral":
        return POSITIVE_EMOJIS[:3] + NEGATIVE_EMOJIS[:3] + AMBIGUOUS_EMOJIS
    return POSITIVE_EMOJIS[:2] + NEUTRAL_EMOJIS[:2] + NEGATIVE_EMOJIS[:2] + AMBIGUOUS_EMOJIS


def sample_emoji(
    sentiment: str,
    rng: random.Random,
    max_count: int = 2,
    mode: str = "aligned",
) -> str:
    if sentiment == "Positive":
        class_pool = POSITIVE_EMOJIS
    elif sentiment == "Neutral":
        class_pool = NEUTRAL_EMOJIS
    elif sentiment == "Negative":
        class_pool = NEGATIVE_EMOJIS
    else:
        class_pool = SARCASM_EMOJIS

    if mode == "shared":
        pool = AMBIGUOUS_EMOJIS + class_pool[:2]
    elif mode == "contradictory":
        pool = contradictory_pool(sentiment)
    else:
        pool = class_pool + AMBIGUOUS_EMOJIS[:3]

    if sentiment == "Neutral":
        count = 1 if rng.random() < 0.9 else 2
    elif sentiment == "Sarcasm":
        count = 1
    elif mode == "contradictory":
        count = 1 if rng.random() < 0.85 else 2
    else:
        count = 1 if rng.random() < 0.75 else min(max_count, 2 + (1 if rng.random() < 0.08 else 0))
    return "".join(rng.choice(pool) for _ in range(count))


def sampled_emoji_mode(rng: random.Random, contradiction_weight: float = 0.0) -> str:
    roll = rng.random()
    if roll < contradiction_weight:
        return "contradictory"
    if roll < contradiction_weight + 0.42:
        return "shared"
    return "aligned"


def maybe_add_punctuation(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.08:
        return text + "!"
    if roll < 0.12:
        return text + "..."
    return text


def stretch_token(token: str, rng: random.Random) -> str:
    positions = [idx for idx, char in enumerate(token) if char.isalpha()]
    if not positions:
        return token
    idx = rng.choice(positions)
    repeats = rng.randint(1, 4)
    return token[: idx + 1] + (token[idx] * repeats) + token[idx + 1 :]


def typo_token(token: str, rng: random.Random) -> str:
    if len(token) < 4:
        return token
    mode = rng.choice(["swap", "drop", "duplicate"])
    if mode == "swap":
        idx = rng.randint(0, len(token) - 2)
        chars = list(token)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if mode == "drop":
        idx = rng.randint(1, len(token) - 2)
        return token[:idx] + token[idx + 1 :]
    idx = rng.randint(1, len(token) - 2)
    return token[: idx + 1] + token[idx] + token[idx + 1 :]


def stylize_text(text: str, rng: random.Random, changes: int) -> str:
    tokens = text.split()
    candidates = [idx for idx, token in enumerate(tokens) if LETTER_RE.search(token)]
    if not candidates:
        return text
    rng.shuffle(candidates)
    for idx in candidates[:changes]:
        if rng.random() < 0.55:
            tokens[idx] = stretch_token(tokens[idx], rng)
        else:
            tokens[idx] = typo_token(tokens[idx], rng)
    return " ".join(tokens)


def attach_emoji(text: str, emoji: str, rng: random.Random) -> str:
    base = maybe_add_punctuation(text, rng)
    if rng.random() < 0.1 and len(base.split()) > 3:
        parts = base.split()
        insert_at = rng.randint(1, len(parts) - 2)
        parts.insert(insert_at, emoji)
        return " ".join(parts)
    return f"{base} {emoji}".strip()


def make_positive_direct(parent: SeedRow, rng: random.Random, noisy: bool) -> dict[str, object]:
    lead = sample_lead("Positive", rng, parent)
    descriptor = sample_descriptor("Positive", rng)
    pattern = rng.choice(
        [
            "{lead} this is {descriptor}",
            "{lead} this is {intensifier} {descriptor}",
            "{lead} this was {descriptor}",
            "{lead} this feels {descriptor}",
        ]
    )
    text = pattern.format(lead=lead, descriptor=descriptor, intensifier=rng.choice(INTENSIFIERS))
    if noisy:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    mode = sampled_emoji_mode(rng)
    comment = attach_emoji(text, sample_emoji("Positive", rng, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "orthographic_noise" if noisy else ("shared_emoji_overlap" if mode == "shared" else "lexical_swap"),
        "phrase_family": "positive_direct",
        "hard_case_flag": 1 if mode == "shared" else 0,
        "sarcasm_flag": 0,
    }


def make_positive_mixed(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Positive", rng, parent)
    descriptor = sample_descriptor("Positive", rng)
    neutral = rng.choice(NEUTRAL_STEMS)
    pattern = rng.choice(
        [
            "{lead} it was {neutral} at first but this is {descriptor}",
            "{lead} i thought it was {neutral} but this is {descriptor}",
            "{lead} it seemed {neutral} at first but this is {descriptor}",
        ]
    )
    text = pattern.format(lead=lead, neutral=neutral, descriptor=descriptor)
    mode = sampled_emoji_mode(rng)
    comment = attach_emoji(text, sample_emoji("Positive", rng, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "mixed_resolution" if mode != "shared" else "shared_emoji_overlap",
        "phrase_family": "positive_mixed_resolution",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_positive_overlap(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Positive", rng, parent)
    text = rng.choice(SHORT_OVERLAP_PHRASES["Positive"]).format(lead=lead)
    if rng.random() < 0.35:
        text = stylize_text(text, rng, changes=1)
    mode = "shared" if rng.random() < 0.8 else "contradictory"
    comment = attach_emoji(text, sample_emoji("Positive", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "short_phrase_overlap" if mode == "shared" else "contradictory_emoji",
        "phrase_family": "positive_short_overlap",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_positive_contradiction(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Positive", rng, parent)
    descriptor = sample_descriptor("Positive", rng)
    text = rng.choice(CONTRADICTION_PATTERNS["Positive"]).format(lead=lead, descriptor=descriptor)
    if rng.random() < 0.4:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    comment = attach_emoji(text, sample_emoji("Positive", rng, max_count=1, mode="contradictory"), rng)
    return {
        "comment": comment,
        "generation_signal": "contradictory_emoji",
        "phrase_family": "positive_contradiction",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_neutral_direct(parent: SeedRow, rng: random.Random, noisy: bool) -> dict[str, object]:
    lead = sample_lead("Neutral", rng, parent)
    descriptor = sample_descriptor("Neutral", rng)
    pattern = rng.choice(
        [
            "{lead} it was {descriptor}",
            "{lead} it was just {descriptor}",
            "{lead} it was kinda {descriptor}",
            "{lead} it felt {descriptor}",
        ]
    )
    text = pattern.format(lead=lead, descriptor=descriptor)
    if noisy:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    mode = sampled_emoji_mode(rng, contradiction_weight=0.08)
    comment = attach_emoji(text, sample_emoji("Neutral", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": (
            "orthographic_noise"
            if noisy
            else ("contradictory_emoji" if mode == "contradictory" else ("shared_emoji_overlap" if mode == "shared" else "lexical_swap"))
        ),
        "phrase_family": "neutral_direct",
        "hard_case_flag": 1 if mode != "aligned" else 0,
        "sarcasm_flag": 0,
    }


def make_neutral_hedged(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Neutral", rng, parent)
    descriptor = sample_descriptor("Neutral", rng)
    pattern = rng.choice(
        [
            "{lead} it was {softener} {descriptor}",
            "{lead} it was {descriptor} i guess",
            "{lead} it was {descriptor} overall",
        ]
    )
    text = pattern.format(lead=lead, softener=rng.choice(SOFTENERS), descriptor=descriptor)
    mode = sampled_emoji_mode(rng, contradiction_weight=0.1)
    comment = attach_emoji(text, sample_emoji("Neutral", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "hedged_neutral" if mode == "aligned" else ("contradictory_emoji" if mode == "contradictory" else "shared_emoji_overlap"),
        "phrase_family": "neutral_hedged",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_neutral_balanced(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Neutral", rng, parent)
    pattern = rng.choice(
        [
            "{lead} not bad not great",
            "{lead} it was okay i guess",
            "{lead} it was fine overall",
            "{lead} it was standard for me",
            "{lead} it was normal overall",
        ]
    )
    text = pattern.format(lead=lead)
    mode = sampled_emoji_mode(rng, contradiction_weight=0.12)
    comment = attach_emoji(text, sample_emoji("Neutral", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "balanced_neutral" if mode == "aligned" else ("contradictory_emoji" if mode == "contradictory" else "shared_emoji_overlap"),
        "phrase_family": "neutral_balanced",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_neutral_overlap(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Neutral", rng, parent)
    text = rng.choice(SHORT_OVERLAP_PHRASES["Neutral"]).format(lead=lead)
    if rng.random() < 0.35:
        text = stylize_text(text, rng, changes=1)
    mode = "shared" if rng.random() < 0.7 else "contradictory"
    comment = attach_emoji(text, sample_emoji("Neutral", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "short_phrase_overlap" if mode == "shared" else "contradictory_emoji",
        "phrase_family": "neutral_short_overlap",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_neutral_contradiction(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Neutral", rng, parent)
    descriptor = sample_descriptor("Neutral", rng)
    text = rng.choice(CONTRADICTION_PATTERNS["Neutral"]).format(lead=lead, descriptor=descriptor)
    if rng.random() < 0.4:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    comment = attach_emoji(text, sample_emoji("Neutral", rng, max_count=1, mode="contradictory"), rng)
    return {
        "comment": comment,
        "generation_signal": "contradictory_emoji",
        "phrase_family": "neutral_contradiction",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_negative_direct(parent: SeedRow, rng: random.Random, noisy: bool) -> dict[str, object]:
    lead = sample_lead("Negative", rng, parent)
    descriptor = sample_descriptor("Negative", rng)
    pattern = rng.choice(
        [
            "{lead} this is {descriptor}",
            "{lead} this is {intensifier} {descriptor}",
            "{lead} this feels {descriptor}",
            "{lead} this turned out {descriptor}",
        ]
    )
    text = pattern.format(lead=lead, descriptor=descriptor, intensifier=rng.choice(INTENSIFIERS))
    if noisy:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    mode = sampled_emoji_mode(rng)
    comment = attach_emoji(text, sample_emoji("Negative", rng, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "orthographic_noise" if noisy else ("shared_emoji_overlap" if mode == "shared" else "lexical_swap"),
        "phrase_family": "negative_direct",
        "hard_case_flag": 1 if mode == "shared" else 0,
        "sarcasm_flag": 0,
    }


def make_negative_mixed(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Negative", rng, parent)
    descriptor = sample_descriptor("Negative", rng)
    neutral = rng.choice(NEUTRAL_STEMS)
    pattern = rng.choice(
        [
            "{lead} it looked {neutral} at first but this is {descriptor}",
            "{lead} i thought it was {neutral} but this is {descriptor}",
            "{lead} it seemed {neutral} at first but this is {descriptor}",
        ]
    )
    text = pattern.format(lead=lead, neutral=neutral, descriptor=descriptor)
    mode = sampled_emoji_mode(rng)
    comment = attach_emoji(text, sample_emoji("Negative", rng, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "mixed_resolution" if mode != "shared" else "shared_emoji_overlap",
        "phrase_family": "negative_mixed_resolution",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_negative_overlap(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Negative", rng, parent)
    text = rng.choice(SHORT_OVERLAP_PHRASES["Negative"]).format(lead=lead)
    if rng.random() < 0.35:
        text = stylize_text(text, rng, changes=1)
    mode = "shared" if rng.random() < 0.78 else "contradictory"
    comment = attach_emoji(text, sample_emoji("Negative", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "short_phrase_overlap" if mode == "shared" else "contradictory_emoji",
        "phrase_family": "negative_short_overlap",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_negative_contradiction(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    lead = sample_lead("Negative", rng, parent)
    descriptor = sample_descriptor("Negative", rng)
    text = rng.choice(CONTRADICTION_PATTERNS["Negative"]).format(lead=lead, descriptor=descriptor)
    if rng.random() < 0.4:
        text = stylize_text(text, rng, changes=rng.randint(1, 2))
    comment = attach_emoji(text, sample_emoji("Negative", rng, max_count=1, mode="contradictory"), rng)
    return {
        "comment": comment,
        "generation_signal": "contradictory_emoji",
        "phrase_family": "negative_contradiction",
        "hard_case_flag": 1,
        "sarcasm_flag": 0,
    }


def make_sarcastic_negative(parent: SeedRow, rng: random.Random, noisy: bool) -> dict[str, object]:
    frame = rng.choice(SARCASM_FRAMES)
    tail = rng.choice(SARCASM_TAILS)
    text = f"{frame} {tail}"
    if noisy:
        text = stylize_text(text, rng, changes=rng.randint(1, 3))
    mode = "shared" if rng.random() < 0.45 else "aligned"
    comment = attach_emoji(text, sample_emoji("Sarcasm", rng, max_count=1, mode=mode), rng)
    return {
        "comment": comment,
        "generation_signal": "sarcasm_negative" if mode == "aligned" else "shared_emoji_overlap",
        "phrase_family": "negative_sarcastic",
        "hard_case_flag": 1,
        "sarcasm_flag": 1,
    }


def choose_positive_variant(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    roll = rng.random()
    if roll < 0.24:
        return make_positive_direct(parent, rng, noisy=False)
    if roll < 0.43:
        return make_positive_direct(parent, rng, noisy=True)
    if roll < 0.63:
        return make_positive_mixed(parent, rng)
    if roll < 0.82:
        return make_positive_overlap(parent, rng)
    return make_positive_contradiction(parent, rng)


def choose_neutral_variant(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    roll = rng.random()
    if roll < 0.2:
        return make_neutral_direct(parent, rng, noisy=False)
    if roll < 0.38:
        return make_neutral_direct(parent, rng, noisy=True)
    if roll < 0.56:
        return make_neutral_hedged(parent, rng)
    if roll < 0.72:
        return make_neutral_balanced(parent, rng)
    if roll < 0.88:
        return make_neutral_overlap(parent, rng)
    return make_neutral_contradiction(parent, rng)


def choose_negative_variant(parent: SeedRow, rng: random.Random) -> dict[str, object]:
    roll = rng.random()
    if roll < 0.22:
        return make_negative_direct(parent, rng, noisy=False)
    if roll < 0.41:
        return make_negative_direct(parent, rng, noisy=True)
    if roll < 0.61:
        return make_negative_mixed(parent, rng)
    if roll < 0.82:
        return make_negative_overlap(parent, rng)
    return make_negative_contradiction(parent, rng)


def split_from_group(group: str) -> str:
    bucket = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def make_record(
    *,
    record_id: str,
    parent: SeedRow,
    comment: str,
    sentiment: str,
    source_type: str,
    generation_signal: str,
    phrase_family: str,
    sarcasm_flag: int,
    hard_case_flag: int,
) -> dict[str, object]:
    split_group = parent.seed_id
    return {
        "id": record_id,
        "parent_id": parent.seed_id,
        "comment": comment,
        "sentiment": sentiment,
        "original_seed_label": parent.original_label,
        "sarcasm_flag": sarcasm_flag,
        "hard_case_flag": hard_case_flag,
        "source_type": source_type,
        "generation_signal": generation_signal,
        "phrase_family": phrase_family,
        "split_group": split_group,
        "recommended_split": split_from_group(split_group),
    }


def export_seed_row(seed: SeedRow, source_partition: str) -> dict[str, object]:
    return {
        "seed_id": seed.seed_id,
        "text": seed.comment,
        "label": seed.sentiment,
        "original_label": seed.original_label,
        "sarcasm_flag": seed.sarcasm_flag,
        "canonical_lead": seed.canonical_lead,
        "source_partition": source_partition,
    }


def build_dataset() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_rows = load_seed_rows()
    generation_seed_rows, holdout_rows, seed_partition_summary = partition_seed_rows(seed_rows, rng)
    holdout_comment_set = {normalize_comment(row.comment) for row in holdout_rows}
    used_comments: set[str] = set(holdout_comment_set)
    records: list[dict[str, object]] = []
    class_counts = Counter()
    sarcasm_counts = Counter()

    for idx, seed in enumerate(generation_seed_rows, start=1):
        normalized = normalize_comment(seed.comment)
        if normalized in used_comments:
            continue
        used_comments.add(normalized)
        record = make_record(
            record_id=f"real_{idx:05d}",
            parent=seed,
            comment=seed.comment,
            sentiment=seed.sentiment,
            source_type="real_seed",
            generation_signal="real_observation",
            phrase_family="real_seed",
            sarcasm_flag=seed.sarcasm_flag,
            hard_case_flag=seed.sarcasm_flag,
        )
        records.append(record)
        class_counts[seed.sentiment] += 1
        sarcasm_counts[(seed.sentiment, seed.sarcasm_flag)] += 1

    positive_rows = [row for row in generation_seed_rows if row.sentiment == "Positive"]
    neutral_rows = [row for row in generation_seed_rows if row.sentiment == "Neutral"]
    negative_rows = [row for row in generation_seed_rows if row.sentiment == "Negative" and row.sarcasm_flag == 0]
    sarcastic_rows = [row for row in generation_seed_rows if row.sarcasm_flag == 1]

    positive_sampler = ParentSampler(positive_rows, rng)
    neutral_sampler = ParentSampler(neutral_rows, rng)
    negative_sampler = ParentSampler(negative_rows, rng)
    sarcastic_sampler = ParentSampler(sarcastic_rows, rng)

    next_id = len(records) + 1
    negative_plain_target = CLASS_TARGETS["Negative"] - NEGATIVE_SARCASM_TARGET

    def append_generated(
        parent: SeedRow,
        generated: dict[str, object],
        sentiment: str,
    ) -> bool:
        nonlocal next_id
        normalized = normalize_comment(str(generated["comment"]))
        if normalized in used_comments:
            return False
        used_comments.add(normalized)
        record = make_record(
            record_id=f"aug_{next_id:05d}",
            parent=parent,
            comment=str(generated["comment"]),
            sentiment=sentiment,
            source_type="controlled_augmentation",
            generation_signal=str(generated["generation_signal"]),
            phrase_family=str(generated["phrase_family"]),
            sarcasm_flag=int(generated["sarcasm_flag"]),
            hard_case_flag=int(generated["hard_case_flag"]),
        )
        records.append(record)
        next_id += 1
        class_counts[sentiment] += 1
        sarcasm_counts[(sentiment, int(generated["sarcasm_flag"]))] += 1
        return True

    def fill_until(
        *,
        sentiment: str,
        target_count: int,
        sampler: ParentSampler,
        generator,
    ) -> None:
        attempts = 0
        while class_counts[sentiment] < target_count:
            parent = sampler.next()
            generated = generator(parent, rng)
            if append_generated(parent, generated, sentiment):
                attempts = 0
                continue
            attempts += 1
            if attempts > 20_000:
                raise RuntimeError(f"Stuck generating unique rows for {sentiment}.")

    def fill_negative_subgroup(
        *,
        sarcasm_flag: int,
        target_count: int,
        sampler: ParentSampler,
        generator,
    ) -> None:
        attempts = 0
        while sarcasm_counts[("Negative", sarcasm_flag)] < target_count:
            parent = sampler.next()
            generated = generator(parent, rng)
            if append_generated(parent, generated, "Negative"):
                attempts = 0
                continue
            attempts += 1
            if attempts > 20_000:
                subgroup = "sarcastic" if sarcasm_flag else "plain"
                raise RuntimeError(f"Stuck generating unique {subgroup} negative rows.")

    fill_until(
        sentiment="Positive",
        target_count=CLASS_TARGETS["Positive"],
        sampler=positive_sampler,
        generator=choose_positive_variant,
    )
    fill_until(
        sentiment="Neutral",
        target_count=CLASS_TARGETS["Neutral"],
        sampler=neutral_sampler,
        generator=choose_neutral_variant,
    )
    fill_negative_subgroup(
        sarcasm_flag=1,
        target_count=NEGATIVE_SARCASM_TARGET,
        sampler=sarcastic_sampler,
        generator=lambda parent, local_rng: make_sarcastic_negative(parent, local_rng, noisy=local_rng.random() < 0.42),
    )
    fill_negative_subgroup(
        sarcasm_flag=0,
        target_count=negative_plain_target,
        sampler=negative_sampler,
        generator=choose_negative_variant,
    )

    if len(records) != TOTAL_ROWS:
        raise RuntimeError(f"Expected {TOTAL_ROWS} rows, found {len(records)}.")

    generation_seed_exports = [
        export_seed_row(seed, source_partition="generation_seed")
        for seed in generation_seed_rows
    ]
    holdout_exports = [
        export_seed_row(seed, source_partition="holdout")
        for seed in holdout_rows
    ]

    manifest = {
        "dataset_name": "social_media_sentiment_dataset_v4",
        "seed_file": str(INPUT_CSV.name),
        "generation_seed_csv": str(GENERATION_SEED_CSV.relative_to(REPO_ROOT)),
        "real_holdout_csv": str(REAL_HOLDOUT_CSV.relative_to(REPO_ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "random_seed": RANDOM_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "target_total_rows": TOTAL_ROWS,
        "class_targets": CLASS_TARGETS,
        "label_policy": {
            "main_labels": ["Negative", "Neutral", "Positive"],
            "sarcastic_mapping": {
                "seed_label": "Sarcastic",
                "mapped_sentiment": "Negative",
                "sarcasm_flag": 1,
                "reason": (
                    "Sarcastic rows are kept in the three-class sentiment scheme by assigning the underlying negative intent "
                    "to the main label while preserving sarcasm as metadata."
                ),
            },
        },
        "real_seed_partition": {
            "total_seed_rows": len(seed_rows),
            "generation_seed_rows": len(generation_seed_rows),
            "holdout_rows": len(holdout_rows),
            "by_original_label": seed_partition_summary,
            "holdout_usage_rule": (
                "Holdout parent rows are excluded from training data generation. Neither the original holdout rows nor their "
                "derived variants appear in the main hybrid dataset."
            ),
        },
        "generation_signal_reasoning": GENERATION_SIGNAL_REASONING,
        "class_counts": dict(Counter(record["sentiment"] for record in records)),
        "source_type_counts": dict(Counter(record["source_type"] for record in records)),
        "generation_signal_counts": dict(Counter(record["generation_signal"] for record in records)),
        "real_partition_counts": {
            "generation_seed": len(generation_seed_exports),
            "holdout": len(holdout_exports),
        },
        "holdout_comment_overlap_with_training_dataset": len(
            {
                normalize_comment(str(record["comment"]))
                for record in records
            }
            & holdout_comment_set
        ),
        "sarcasm_breakdown": {
            "negative_sarcastic_rows": sarcasm_counts[("Negative", 1)],
            "negative_plain_rows": sarcasm_counts[("Negative", 0)],
        },
        "split_counts": dict(Counter(record["recommended_split"] for record in records)),
        "hard_case_rows": sum(int(record["hard_case_flag"]) for record in records),
        "unique_comments": len({normalize_comment(str(record["comment"])) for record in records}),
    }

    return records, generation_seed_exports, holdout_exports, manifest


def write_outputs(
    records: list[dict[str, object]],
    generation_seed_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    fieldnames = [
        "id",
        "parent_id",
        "comment",
        "sentiment",
        "original_seed_label",
        "sarcasm_flag",
        "hard_case_flag",
        "source_type",
        "generation_signal",
        "phrase_family",
        "split_group",
        "recommended_split",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    seed_fieldnames = [
        "seed_id",
        "text",
        "label",
        "original_label",
        "sarcasm_flag",
        "canonical_lead",
        "source_partition",
    ]
    with GENERATION_SEED_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fieldnames)
        writer.writeheader()
        writer.writerows(generation_seed_rows)

    with REAL_HOLDOUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fieldnames)
        writer.writeheader()
        writer.writerows(holdout_rows)

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    records, generation_seed_rows, holdout_rows, manifest = build_dataset()
    write_outputs(records, generation_seed_rows, holdout_rows, manifest)
    print(f"Wrote {len(records)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(generation_seed_rows)} generation seeds to {GENERATION_SEED_CSV}")
    print(f"Wrote {len(holdout_rows)} untouched holdout rows to {REAL_HOLDOUT_CSV}")
    print(json.dumps(manifest["class_counts"], indent=2))
    print(json.dumps(manifest["generation_signal_counts"], indent=2))


if __name__ == "__main__":
    main()
