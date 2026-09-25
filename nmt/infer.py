"""Test d'inférence rapide du modèle fine-tuné fr↔ewo.

Charge NLLB-200 + adaptateur LoRA + tokenizer étendu (avec ewo_Latn),
puis génère quelques traductions dans les deux directions.

Usage :
    python infer.py --adapter outputs/nllb-ewondo-lora
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-600M"
NEW_LANG = "ewo_Latn"
FR_LANG = "fra_Latn"

TESTS_FR = [
    "Bonjour, comment allez-vous ?",
    "La Bible répond à ces questions importantes.",
    "Nous nous posons tous des questions sur la vie.",
    "Beaucoup de personnes ont constaté que la Bible est utile.",
]

TESTS_EWO = [
    "Mbolo, wa yem?",
    "Bibel a yalen minsili mi.",
]


def translate(model, tokenizer, text: str, src_lang: str, tgt_lang: str,
              max_length: int = 128) -> str:
    src_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    eos_id = tokenizer.eos_token_id

    text_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=max_length - 2
    )
    input_ids = torch.tensor([[src_id] + text_ids + [eos_id]]).to(model.device)

    output = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=tgt_id,
        max_length=max_length,
        num_beams=4,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    args = ap.parse_args()

    print(f"Chargement du tokenizer depuis {args.adapter}…")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=False)

    print(f"Chargement du modèle de base {BASE_MODEL}…")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"Application de l'adaptateur LoRA…")
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Modèle sur GPU.")
    else:
        print("Modèle sur CPU (inférence plus lente).")

    print("\n=== FR → EWO ===")
    for fr in TESTS_FR:
        with torch.no_grad():
            out = translate(model, tokenizer, fr, FR_LANG, NEW_LANG)
        print(f"FR : {fr}")
        print(f"EWO: {out}\n")

    print("=== EWO → FR ===")
    for ewo in TESTS_EWO:
        with torch.no_grad():
            out = translate(model, tokenizer, ewo, NEW_LANG, FR_LANG)
        print(f"EWO: {ewo}")
        print(f"FR : {out}\n")


if __name__ == "__main__":
    main()
