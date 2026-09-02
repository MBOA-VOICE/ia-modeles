# TTS — Synthèse vocale

Module **partagé** entre `mboa_voice` et `mboa_culture`.

## Modèle de base

- **Coqui TTS / XTTS** — fork communautaire maintenu : `idiap/coqui-ai-TTS`
  (package pip : `coqui-tts`). La société Coqui a fermé en 2024 mais la
  librairie reste open source.
- XTTS v2 supporte le voice cloning avec quelques secondes de référence.

## Approche

Fine-tuning par langue à partir d'enregistrements de locuteurs natifs
(corpus dans `../data/{langue}/`).

## Convention de nommage HF

`langues-camerounaises/xtts-{langue}`
