# Sources publiques par langue — bootstrap corpus

Recensement de sources en accès libre pour amorcer les corpus des 5
langues cibles **sans dépendre** de la collecte terrain des étudiantes.
Objectif : disposer d'un jeu de données de base immédiatement
utilisable, complété au fil de l'eau par les données collectées.

## Résumé exécutif

- **JW.org** couvre **les 5 langues cibles**, texte + audio, aligné
  verset par verset avec le français (via la Bible et les brochures).
  → **source principale** pour bootstrap ASR + NMT + TTS.
- **NLLB-200** ne connaît nativement que le **Fulfulde** (`fuv_Latn`)
  parmi nos 5. Ewondo / Douala / Fe'efe'e / Ghomala' devront être
  ajoutés au modèle par fine-tuning (voir [nmt/](../nmt/)).
- **Whisper** est multilingue mais pas entraîné sur nos langues.
  Fonctionne néanmoins en mode transcription phonétique
  approximative en attendant fine-tuning.

## Couverture NLLB-200 (traduction publique out-of-the-box)

| Langue    | Code NLLB     | Support natif |
|-----------|---------------|---------------|
| Fulfulde  | `fuv_Latn`    | ✓ oui         |
| Ewondo    | —             | ✗ fine-tuning requis |
| Douala    | —             | ✗ fine-tuning requis |
| Fe'efe'e  | —             | ✗ fine-tuning requis |
| Ghomala'  | —             | ✗ fine-tuning requis |

## JW.org — codes langue et URL de base

| Langue    | Code JW  | Libellé JW.org       |
|-----------|----------|----------------------|
| Ewondo    | `EWN`    | Ewondo               |
| Douala    | `DA`     | Douala               |
| Fulfulde  | (var.)   | Fulfulde (Cameroun)  |
| Fe'efe'e  | `FFE`    | Bafang               |
| Ghomala'  | `GHM`    | Bandjoun-Baham       |

Contenus attendus par langue :
- **Texte** : Nouveau Testament (traduction Monde Nouveau) + brochures
  d'enseignement, tous alignés avec la version FR.
- **Audio** : lectures audio du NT (voix off professionnelle),
  identifiées verset par verset → parfait pour ASR + TTS.

Point d'entrée technique : API `pubMedia` de jw.org, endpoint public
non authentifié
(`https://www.jw.org/en/library/bible/study-bible/books/json/{lang}/{book}/{chapter}`
et variantes). Voir le repo tiers `MrCyjaneK/jwapi` pour la structure
de réponse.

**Robots.txt** : `/library/` non bloqué, pas de crawl-delay explicite
→ scraping possible en throttling raisonnable (1 req/s). Respecter les
conditions d'usage (usage éducatif / recherche : OK).

## Autres sources par langue

### Fulfulde
- **WaxalNLP** (Google) sur HF Datasets — 101k phrases ASR + 3.88k
  échantillons TTS. Utilise le code `ful`.
  URL : `huggingface.co/datasets/google/WaxalNLP`
- **Common Voice** (Mozilla) — quelques heures en Fulfulde nigérian.

### Ewondo, Douala, Fe'efe'e, Ghomala'
- Aucun dataset HF ou OpenSLR notable de taille significative
  (état sept. 2026). Bootstrap = JW.org quasi exclusif.
- **YouVersion** (`bible.com/languages/ewo` etc.) — bibles alternatives
  en texte, utile pour aligner avec traductions non-JW si dispo.
- **Wikipedia** — versions en Ewondo (`ewo.wikipedia.org`) et Douala
  (`dua.wikipedia.org`) existent mais très petites (< 100 articles).

### Ressources générales
- **AfriHuBERT** — modèle self-supervised pour langues africaines,
  pré-entraînement utile comme point de départ ASR.
- **NLLB seed data** — Meta publie les jeux d'amorçage NLLB, utile pour
  comprendre la structure attendue par NLLB lors du fine-tuning.

## Ce que ça change pour la stratégie

1. Le **corpus parallèle FR↔langue** de la Tâche 4 (Samuela) peut être
   pré-amorcé à ~5000-10000 paires par langue via JW.org, **avant même**
   qu'elle collecte quoi que ce soit. Sa contribution ajoute du contenu
   *contemporain* et *non-religieux* que JW.org ne peut pas fournir.
2. Le **corpus audio** de la Tâche 3 démarre avec les lectures audio
   JW.org (voix off pro, très propre, > 20h par langue). Les
   enregistrements des étudiantes ajoutent de la *diversité de voix*
   et de *l'accent naturel* que la voix off n'a pas.
3. Le **fine-tuning NLLB** peut démarrer dès que le scraper JW.org
   tourne — pas besoin d'attendre les étudiantes.

## Voir aussi

- [../README.md](../README.md) — répartition étudiantes ↔ langues
- [../nmt/README.md](../nmt/README.md) — approche fine-tuning NLLB
- [../asr/README.md](../asr/README.md) — approche fine-tuning Whisper
