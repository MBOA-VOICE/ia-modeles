"""Prépare le dataset fr↔ewo à partir du corpus JW pour fine-tuning NLLB.

Le CSV source contient des paragraphes entiers (1 ligne = plusieurs
phrases côté fr et côté ewo). On tente un découpage phrase-à-phrase :
si les deux côtés ont le même nombre de phrases, on émet N paires ;
sinon on garde le paragraphe tel quel (filtré ensuite par max_length
au tokenizer).

Sortie : ia-modeles/nmt/datasets/ewondo/ (format HuggingFace arrow).

Usage :
    python prepare_dataset.py \\
        --input ../data/ewondo/parallel_jw/lff_fr-ewo.csv \\
        --output datasets/ewondo
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import Dataset, DatasetDict

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÉÈÀÂÊÎÔÛÇ0-9«“\"'BbEeIiMmNnOoAa])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return parts or [text]


def build_pairs(csv_path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, tgt = row["source_text"], row["target_text"]
            src_sents = split_sentences(src)
            tgt_sents = split_sentences(tgt)

            if len(src_sents) == len(tgt_sents) and len(src_sents) > 1:
                for s, t in zip(src_sents, tgt_sents):
                    pairs.append({"fr": s, "ewo": t})
            else:
                pairs.append({"fr": src.strip(), "ewo": tgt.strip()})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = build_pairs(args.input)
    print(f"Paires extraites : {len(pairs)}")

    ds = Dataset.from_list(pairs).shuffle(seed=args.seed)
    n_val = max(1, int(len(ds) * args.val_ratio))
    dsd = DatasetDict({
        "train": ds.select(range(n_val, len(ds))),
        "validation": ds.select(range(n_val)),
    })
    dsd.save_to_disk(str(args.output))
    print(f"Dataset sauvegardé dans {args.output}")
    print(f"  train:      {len(dsd['train'])}")
    print(f"  validation: {len(dsd['validation'])}")


if __name__ == "__main__":
    main()
