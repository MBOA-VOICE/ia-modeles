# Évaluation de prononciation

Module **partagé** entre `mboa_voice` (module d'apprentissage gamifié) et
`mboa_culture` (feedback dans les leçons).

## Principe (MVP)

Basé sur ASR : l'apprenant prononce une phrase cible, on transcrit avec le
Whisper fine-tuné (`../asr/`), et on calcule un score de similarité
(WER / distance au niveau phonème) entre la transcription et la cible.

## Extensions possibles

- **GOP** (Goodness of Pronunciation) via `wav2vec2` : score par phonème.
- **Forced alignment** (Montreal Forced Aligner, `aeneas`) : détecter les
  segments mal prononcés au sein d'une phrase.

Ces extensions ne sont pas nécessaires pour la soutenance — le MVP basé
texte est suffisant pour démontrer le concept.
