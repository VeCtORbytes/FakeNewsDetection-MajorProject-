"""
Dataset class for multilingual fake news detection.
Supports XLM-RoBERTa, MuRIL, and the ensemble pipeline.

Key improvements:
    - token_type_ids included when the tokenizer provides them (MuRIL/BERT).
    - Tokenizer loaded once at init with use_fast=True.
    - Input validation for mismatched lengths or empty datasets.
    - Texts cast to str and labels cast to int up front.
    - tokenizer_name stored for downstream inspection.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


TOKENIZER_ALIASES = {
    "ensemble": "google/muril-base-cased",
}


class MultilingualFakeNewsDataset(Dataset):
    """
    PyTorch Dataset for multilingual fake news detection.

    Returns batches compatible with all three model types:
        XLM-RoBERTa  — uses input_ids + attention_mask
        MuRIL        — uses input_ids + attention_mask + token_type_ids
        Ensemble     — uses the MuRIL tokenizer (consistent with inference)

    Each __getitem__ returns a dict with:
        input_ids:       LongTensor [max_length]
        attention_mask:  LongTensor [max_length]
        token_type_ids:  LongTensor [max_length]  — only present when the
                         tokenizer produces it (MuRIL/BERT); absent for
                         XLM-RoBERTa (RoBERTa-style tokenizers omit it)
        labels:          LongTensor scalar
        language:        str  — language code for per-language analysis
    """

    def __init__(
        self,
        texts,
        labels,
        languages,
        tokenizer_name: str,
        max_length: int = 512,
    ):
        """
        Args:
            texts:          Sequence of raw text strings.
            labels:         Sequence of integer labels (0 = fake, 1 = real).
            languages:      Sequence of language code strings (e.g. 'hi', 'en').
            tokenizer_name: HuggingFace tokenizer identifier.
                            'xlm-roberta-base'        for XLM-RoBERTa
                            'google/muril-base-cased' for MuRIL / ensemble
                            'ensemble'                to mirror inference config
            max_length:     Tokenization truncation/padding length.
        """
        if not (len(texts) == len(labels) == len(languages)):
            raise ValueError(
                f"texts ({len(texts)}), labels ({len(labels)}), and "
                f"languages ({len(languages)}) must all have the same length."
            )
        if len(texts) == 0:
            raise ValueError("Dataset is empty — texts list has zero elements.")

        self.texts = [str(t) for t in texts]
        try:
            self.labels = [int(l) for l in labels]
        except (TypeError, ValueError) as exc:
            raise ValueError("All labels must be castable to int.") from exc
        self.languages = list(languages)
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name

        resolved_name = TOKENIZER_ALIASES.get(tokenizer_name, tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name, use_fast=True)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        language = self.languages[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "language": language,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item
