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

## Pipeline de fine-tuning ewondo (v0)

### 1. Préparer le dataset (local ou Colab)

```bash
pip install -r requirements.txt
python prepare_dataset.py \
    --input ../data/ewondo/parallel_jw/lff_fr-ewo.csv \
    --output datasets/ewondo
```

Découpe les paragraphes en phrases quand les comptes fr/ewo correspondent
(sinon garde le paragraphe entier, tronqué au tokenizer). Produit
`datasets/ewondo/{train,validation}/` au format Arrow.

### 2. Lancer l'entraînement (Colab T4 recommandé)

Notebook Colab minimal (à copier dans une cellule) :

```python
!git clone https://github.com/MBOA-VOICE/ia-modeles.git
%cd ia-modeles/nmt
!pip install -q -r requirements.txt

# uploader datasets/ewondo/ dans Colab, puis :
!python train_ewondo.py \
    --dataset datasets/ewondo \
    --output outputs/nllb-ewondo-lora \
    --epochs 3
```

Durée attendue : **2 à 4 h** sur T4 avec LoRA (rank 16, batch effectif 16).
Sur CPU c'est ~50-100× plus lent → ne pas essayer localement.

### 3. Récupérer les poids

Zipper `outputs/nllb-ewondo-lora/` (adaptateur LoRA ~50 Mo) et le pusher
sur HF Hub sous `langues-camerounaises/nllb-ewondo-lora`, ou le copier
dans le backend pour test.

### Notes techniques

- L'ewondo est **ajouté au tokenizer** comme token spécial `ewo_Latn`
  (pas dans NLLB natif). Les embeddings sont redimensionnés en conséquence.
- Entraînement **bidirectionnel** fr↔ewo dans un seul run (direction
  tirée au sort par exemple).
- Corpus actuel : ~3 000 paragraphes JW → ~10-15 k paires phrases après
  split. C'est un plancher, à enrichir dès que possible.
