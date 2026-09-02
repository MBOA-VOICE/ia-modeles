# ASR — Reconnaissance vocale

Module **partagé** entre `mboa_voice` et `mboa_culture` : les deux projets
ont besoin de transcrire de la parole en langues camerounaises.

## Modèle de base

- **Whisper** (OpenAI) — modèle multilingue, base pour le fine-tuning.
- **`faster-whisper`** — port CTranslate2, ~4× plus rapide en CPU, même
  précision. À utiliser pour l'inférence sur le VPS (pas de GPU).

## Approche

Fine-tuner Whisper-small ou Whisper-medium sur chaque langue camerounaise
(corpus dans `../data/{langue}/`). Publier les checkpoints sur Hugging
Face Hub.

## Convention de nommage HF

`langues-camerounaises/whisper-{taille}-{langue}`

Exemples :
- `langues-camerounaises/whisper-small-ewondo`
- `langues-camerounaises/whisper-medium-fulfulde`
