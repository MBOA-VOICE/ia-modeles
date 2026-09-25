# ia-modeles

Repo dédié à l'entraînement et au fine-tuning des modèles IA utilisés par
les projets `mboa_voice` (traduction vocale) et `mboa_culture` (Mboa —
patrimoine culturel).

Séparé du repo `backend/` car outils différents : notebooks Colab/Kaggle,
GPU, cycle expérimental. Le backend n'importe que les modèles finalisés
via Hugging Face Hub.

## Structure

| Dossier | Rôle | Partagé / Propre |
|---|---|---|
| `data/` | Corpus audio et textuels par langue | Partagé (chaque étudiante collecte pour les langues qui lui sont assignées) |
| `asr/` | Reconnaissance vocale (fine-tuning Whisper) | Partagé |
| `tts/` | Synthèse vocale (Coqui XTTS) | Partagé |
| `pronunciation/` | Évaluation de prononciation (basée ASR) | Partagé |
| `nmt/` | Traduction automatique (fine-tuning NLLB) | Propre à `mboa_voice` |

## Poids des modèles

**Ne jamais committer les poids** — fichiers lourds, historique Git
ingérable. Publier sur Hugging Face Hub sous l'organisation associée au
projet. Le backend télécharge les poids au démarrage via `huggingface_hub`.

## Entraînement

Colab (T4 gratuit) ou Kaggle Notebooks (T4/P100 gratuit, sessions plus
longues). Sauvegarder les checkpoints régulièrement — les sessions
gratuites sont limitées en durée.

## Répartition de collecte de corpus

À compléter par les deux étudiantes.

| Langue    | Collectée par | Volume cible | État |
|-----------|---------------|--------------|------|
| Ewondo    | BOLO BOLO          | 2H audio+1000 paires fr->ewondo         | en cours |
| Fulfulde  | TODO          | TODO         | TODO |
| Douala    | TODO          | TODO         | TODO |
| Fe'efe'e  | TODO          | TODO         | TODO |
| Ghomala'  | BOLO BOLO         | 2h audio+500 paires fr->ghomala         | non Commencé |
