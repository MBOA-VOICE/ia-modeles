# NMT — Traduction automatique

Module **propre à `mboa_voice`** uniquement (`mboa_culture` utilise l'API
Claude pour la génération de contenu, pas de la traduction automatique
entraînée).

## Modèle de base

- **NLLB** (Meta — *No Language Left Behind*) : supporte 200 langues,
  dont plusieurs langues africaines. Base pour le fine-tuning sur les
  paires fr/en ↔ langues camerounaises.
- Variantes :
  - `nllb-200-distilled-600M` — léger, CPU-friendly, prioritaire pour le VPS.
  - `nllb-200-3.3B` — meilleure qualité, GPU requis pour l'inférence.

## Approche

Fine-tuning sur paires parallèles fr/en ↔ {ewondo, fulfulde, douala,
fe'efe'e, ghomala'}. Corpus dans `../data/{langue}/`.

## Convention de nommage HF

`langues-camerounaises/nllb-{langue}`
