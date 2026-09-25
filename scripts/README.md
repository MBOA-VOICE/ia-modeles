# scripts/

Utilitaires pour amorcer les corpus à partir de sources publiques.

## `scrape_jw.py`

Télécharge audio + texte des publications jw.org disponibles pour
Ewondo, Douala, Fulfulde et Ghomala' (Fe'efe'e non couvert par l'API).

**Prérequis** : Python 3.10+, aucune dépendance externe (stdlib
uniquement — `urllib`, `json`, `hashlib`).

**Usage**

```bash
# Sec, sans téléchargement — pour vérifier le catalogue :
python scripts/scrape_jw.py --dry-run

# Aspiration complète (audio + brochures), pour les 4 langues :
python scripts/scrape_jw.py

# Seulement Ewondo :
python scripts/scrape_jw.py --lang ewo

# Avec la version FR (pour alignement) :
python scripts/scrape_jw.py --french
```

**Sortie**

```
data/
├── ewondo/jw_org/audio/sjjm_01.mp3, sjjm_02.mp3, …
├── ewondo/jw_org/text/lff.epub, lff.pdf, …
├── douala/jw_org/…
├── fulfulde/jw_org/…
├── ghomala/jw_org/…
├── _french/jw_org/…                (si --french)
└── jw_org_manifest.json            manifeste consolidé (URL, taille, durée)
```

Les dossiers `audio/` et fichiers `*.mp3`, `*.epub`, `*.pdf` sont
gitignorés — trop lourds pour Git. Republication éventuelle sur HF Datasets.

**Respect du site**

- Throttle 0.8 s entre téléchargements par défaut (paramétrable
  `--throttle`).
- User-Agent identifié à un contact e-mail utilisable par jw.org
  en cas de problème.
- Robots.txt vérifié : `/library/` non bloqué, pas de crawl-delay.

## À venir

- `align_jw.py` — extraction texte des EPUB + alignement paragraphe à
  paragraphe FR ↔ langue-cible, pour produire des CSV parallèles
  exploitables par NLLB.
- `upload_hf.py` — publication automatique des corpus produits sur
  l'organisation HuggingFace MBOA-VOICE.
