"""Fine-tuning NLLB-200 pour ajouter l'ewondo (fr↔ewo).

Trois étapes clés :
1. On étend le tokenizer NLLB avec le token de langue « ewo_Latn » (absent
   du vocabulaire natif) et on redimensionne les embeddings du modèle.
2. On configure LoRA (rank 16, projections q/v) pour tenir sur un GPU
   T4 16 Go avec fp16.
3. On entraîne dans les deux directions fr→ewo et ewo→fr.

Cible : Colab T4 gratuit, ~2-4h pour 3 epochs sur ~10k paires.

Usage :
    python train_ewondo.py \\
        --dataset datasets/ewondo \\
        --output outputs/nllb-ewondo-lora
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_MODEL = "facebook/nllb-200-distilled-600M"
NEW_LANG = "ewo_Latn"
FR_LANG = "fra_Latn"


def extend_tokenizer(tokenizer):
    """Ajoute ewo_Latn comme token spécial et met à jour lang_code_to_id.

    Nécessaire car NLLB ne connaît pas l'ewondo. Sans cela, tokenizer.src_lang
    = "ewo_Latn" lèverait une KeyError.
    """
    existing = tokenizer.special_tokens_map.get("additional_special_tokens", []) or []
    if NEW_LANG in existing:
        return tokenizer

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(existing) + [NEW_LANG]}
    )
    new_id = tokenizer.convert_tokens_to_ids(NEW_LANG)
    tokenizer.lang_code_to_id[NEW_LANG] = new_id
    if hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code[new_id] = NEW_LANG
    if hasattr(tokenizer, "fairseq_tokens_to_ids"):
        tokenizer.fairseq_tokens_to_ids[NEW_LANG] = new_id
    if hasattr(tokenizer, "fairseq_ids_to_tokens"):
        tokenizer.fairseq_ids_to_tokens[new_id] = NEW_LANG
    return tokenizer


def make_preprocess(tokenizer, max_length: int):
    def preprocess(batch):
        directions = [random.choice(["fr2ewo", "ewo2fr"]) for _ in batch["fr"]]
        srcs, tgts, src_langs, tgt_langs = [], [], [], []
        for fr, ewo, d in zip(batch["fr"], batch["ewo"], directions):
            if d == "fr2ewo":
                srcs.append(fr); tgts.append(ewo)
                src_langs.append(FR_LANG); tgt_langs.append(NEW_LANG)
            else:
                srcs.append(ewo); tgts.append(fr)
                src_langs.append(NEW_LANG); tgt_langs.append(FR_LANG)

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for src, tgt, sl, tl in zip(srcs, tgts, src_langs, tgt_langs):
            tokenizer.src_lang = sl
            enc = tokenizer(
                src, text_target=tgt, max_length=max_length,
                truncation=True, padding=False,
            )
            enc["input_ids"][0] = tokenizer.convert_tokens_to_ids(sl)
            enc["labels"][0] = tokenizer.convert_tokens_to_ids(tl)
            model_inputs["input_ids"].append(enc["input_ids"])
            model_inputs["attention_mask"].append(enc["attention_mask"])
            model_inputs["labels"].append(enc["labels"])
        return model_inputs
    return preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-lora", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    tokenizer = extend_tokenizer(tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dsd = load_from_disk(str(args.dataset))
    preprocess = make_preprocess(tokenizer, args.max_length)
    tokenized = dsd.map(
        preprocess, batched=True, batch_size=32,
        remove_columns=dsd["train"].column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=args.max_length,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Modèle sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
