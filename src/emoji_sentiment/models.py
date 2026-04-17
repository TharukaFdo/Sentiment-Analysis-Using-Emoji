from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _apply_branch_dropout(
    features: torch.Tensor,
    dropout: float,
    *,
    training: bool,
) -> torch.Tensor:
    if not training or dropout <= 0.0:
        return features
    keep_prob = 1.0 - dropout
    if keep_prob <= 0.0:
        return torch.zeros_like(features)
    mask = torch.bernoulli(
        torch.full(
            (features.size(0), 1),
            keep_prob,
            device=features.device,
            dtype=features.dtype,
        )
    )
    return (features * mask) / keep_prob


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_id: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dropout(features)


class TextSentimentModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        output_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input_ids, lengths)
        return self.classifier(features)


class EmojiSentimentModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        output_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, emoji_ids: torch.Tensor, emoji_lengths: torch.Tensor) -> torch.Tensor:
        features = self.encoder(emoji_ids, emoji_lengths)
        return self.classifier(features)


class CombinedSentimentModel(nn.Module):
    def __init__(
        self,
        *,
        text_vocab_size: int,
        text_pad_id: int,
        emoji_vocab_size: int,
        emoji_pad_id: int,
        output_dim: int,
        dropout: float = 0.35,
        emoji_branch_dropout: float = 0.35,
        emoji_scale: float = 0.45,
        text_fusion_dim: int = 192,
        emoji_fusion_dim: int = 64,
        fusion_hidden_dim: int = 192,
    ):
        super().__init__()
        self.text_encoder = SequenceEncoder(
            vocab_size=text_vocab_size,
            embed_dim=128,
            hidden_dim=128,
            pad_id=text_pad_id,
            dropout=dropout,
        )
        self.emoji_encoder = SequenceEncoder(
            vocab_size=emoji_vocab_size,
            embed_dim=64,
            hidden_dim=96,
            pad_id=emoji_pad_id,
            dropout=dropout,
        )
        self.emoji_branch_dropout = emoji_branch_dropout
        self.emoji_scale = emoji_scale
        self.text_projection = nn.Sequential(
            nn.Linear(128 * 2, text_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emoji_projection = nn.Sequential(
            nn.Linear(96 * 2, emoji_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emoji_gate = nn.Sequential(
            nn.Linear(text_fusion_dim + emoji_fusion_dim, max(64, emoji_fusion_dim)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(64, emoji_fusion_dim), 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(text_fusion_dim + emoji_fusion_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        emoji_ids: torch.Tensor,
        emoji_lengths: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_projection(self.text_encoder(input_ids, lengths))
        emoji_features = self.emoji_projection(self.emoji_encoder(emoji_ids, emoji_lengths))
        emoji_gate = self.emoji_gate(torch.cat([text_features, emoji_features], dim=1))
        emoji_features = emoji_features * emoji_gate
        emoji_features = _apply_branch_dropout(
            emoji_features,
            self.emoji_branch_dropout,
            training=self.training,
        )
        emoji_features = emoji_features * self.emoji_scale
        merged = torch.cat([text_features, emoji_features], dim=1)
        return self.fusion(merged)


def build_models(
    *,
    text_vocab_size: int,
    text_pad_id: int,
    emoji_vocab_size: int,
    emoji_pad_id: int,
    combined_emoji_branch_dropout: float = 0.35,
    combined_emoji_scale: float = 0.45,
) -> dict[str, nn.Module]:
    return {
        "text": TextSentimentModel(
            vocab_size=text_vocab_size,
            pad_id=text_pad_id,
            output_dim=3,
        ),
        "emoji": EmojiSentimentModel(
            vocab_size=emoji_vocab_size,
            pad_id=emoji_pad_id,
            output_dim=3,
        ),
        "combined": CombinedSentimentModel(
            text_vocab_size=text_vocab_size,
            text_pad_id=text_pad_id,
            emoji_vocab_size=emoji_vocab_size,
            emoji_pad_id=emoji_pad_id,
            output_dim=3,
            emoji_branch_dropout=combined_emoji_branch_dropout,
            emoji_scale=combined_emoji_scale,
        ),
    }
