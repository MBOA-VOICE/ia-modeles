# MAFAND-MT — paires parallèles FR ↔ Ghomala'

Corpus **MAFAND-MT** (Masakhane Anglophone/Francophone African News Domain
Machine Translation) — paires parallèles traduites à la main par des
locuteurs Ghomala', domaine des news, non religieux.

## Volumes

| Split | Paires |
|-------|--------|
| train | 2 232  |
| dev   | 1 133  |
| test  | 1 430  |
| **Total** | **4 795** |

Fichiers : `{split}.fr` (français) et `{split}.bbj` (ghomala') alignés
ligne à ligne (ligne N du .fr correspond à ligne N du .bbj).

## Provenance

Récupéré depuis le dépôt officiel Masakhane :
[github.com/masakhane-io/lafand-mt](https://github.com/masakhane-io/lafand-mt)
sous `data/text_files/fr-bbj/`.

Papier de référence : Adelani et al., *"A Few Thousand Translations Go a
Long Way! Leveraging Pre-Trained Models for African News Translation"*,
NAACL 2022.

## Licence

Créative Commons Attribution 4.0 (CC-BY-4.0). Utilisation académique et
commerciale autorisée avec attribution.

## Utilisation

```python
def load_split(name):
    with open(f"{name}.fr") as f_fr, open(f"{name}.bbj") as f_bbj:
        return list(zip(f_fr, f_bbj))

pairs = load_split("train")
print(len(pairs), "paires")
print(pairs[0])
```

## Complémentarité

Ce corpus est **complémentaire** aux sources jw.org (religieuses) et à
la collecte terrain de Samuela. Il apporte du vocabulaire
**contemporain et non-religieux** (politique, sport, actualité) que
jw.org ne peut pas fournir.
