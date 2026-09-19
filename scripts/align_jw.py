#!/usr/bin/env python3
"""
Aligne les EPUB jw.org langue-cible ↔ FR en CSV parallèles pour NLLB.

Principe : jw.org utilise le MÊME identifiant de fichier XHTML entre versions
linguistiques (1102015140.xhtml en DA = même chapitre en FR), et chaque <p>
porte un data-pid stable. On zippe les deux EPUB et on merge sur (fichier, pid).

Usage:
    python scripts/align_jw.py --lang dua --pub bhs
    python scripts/align_jw.py --all

Sortie:
    data/{lang}/parallel_jw/{pub}_fr-{iso}.csv
    format: source_lang,target_lang,source_text,target_text
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NS = "{http://www.w3.org/1999/xhtml}"

# Alignement avec scrape_jw.py
ISO = {"douala": "dua", "ewondo": "ewo", "ghomala": "bbj"}
LANG_BY_ISO = {v: k for k, v in ISO.items()}

PUBS = ["bhs", "jl", "lff"]

# Fichier de chapitre canonique : numérique uniquement, sans suffixe extracted
CHAP_RE = re.compile(r"^(\d+)\.xhtml$")


def iter_chapter_files(epub_path: pathlib.Path):
    """Yield (chapter_id, xhtml_bytes) pour les chapitres canoniques uniquement."""
    with zipfile.ZipFile(epub_path) as z:
        for name in z.namelist():
            base = pathlib.PurePosixPath(name).name
            m = CHAP_RE.match(base)
            if not m:
                continue
            yield m.group(1), z.read(name)


def extract_paragraphs(epub_path: pathlib.Path) -> dict[tuple[str, str], str]:
    """Retourne {(chapter_id, data_pid): texte_nettoyé}."""
    out: dict[tuple[str, str], str] = {}
    for chap, xhtml in iter_chapter_files(epub_path):
        try:
            root = ET.fromstring(xhtml)
        except ET.ParseError:
            continue
        _walk(root, chap, out, in_aside=False)
    return out


def _walk(elem: ET.Element, chap: str, out: dict, in_aside: bool) -> None:
    tag = elem.tag.split("}", 1)[-1]
    if tag == "aside":
        # Les <aside> contiennent les citations bibliques, souvent restées
        # en FR même dans la version langue-cible → on les ignore.
        in_aside = True
    if tag == "p" and not in_aside:
        pid = elem.get("data-pid")
        if pid:
            text = _clean(elem)
            if text:
                out[(chap, pid)] = text
            return
    for child in elem:
        _walk(child, chap, out, in_aside)


def _clean(elem: ET.Element) -> str:
    parts: list[str] = []
    _collect(elem, parts)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _collect(elem: ET.Element, parts: list[str]) -> None:
    tag = elem.tag.split("}", 1)[-1]
    cls = elem.get("class") or ""
    # Ignorer numéros de page et références de note
    if tag == "span" and "pageNum" in cls:
        return
    if tag == "a" and elem.get("{http://www.idpf.org/2007/ops}type") == "noteref":
        return
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        _collect(child, parts)
        if child.tail:
            parts.append(child.tail)


def align_pub(lang_dir: str, pub: str) -> pathlib.Path | None:
    iso = ISO[lang_dir]
    src = REPO_ROOT / "data" / lang_dir / "jw_org" / "text" / f"{pub}.epub"
    fr = REPO_ROOT / "data" / "_french" / "jw_org" / "text" / f"{pub}.epub"
    if not src.exists():
        print(f"  [skip] {lang_dir}/{pub}: {src.relative_to(REPO_ROOT)} absent")
        return None
    if not fr.exists():
        print(f"  [skip] {lang_dir}/{pub}: {fr.relative_to(REPO_ROOT)} absent")
        return None

    src_map = extract_paragraphs(src)
    fr_map = extract_paragraphs(fr)
    common = sorted(set(src_map) & set(fr_map), key=lambda k: (int(k[0]), int(k[1])))

    rows: list[tuple[str, str, str, str]] = []
    for key in common:
        s_text, f_text = src_map[key], fr_map[key]
        # Paires substantielles : au moins 3 mots des deux côtés
        if len(s_text.split()) < 3 or len(f_text.split()) < 3:
            continue
        rows.append(("fr", iso, f_text, s_text))

    out_dir = REPO_ROOT / "data" / lang_dir / "parallel_jw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pub}_fr-{iso}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_lang", "target_lang", "source_text", "target_text"])
        w.writerows(rows)

    print(f"  [ok] {lang_dir}/{pub}: {len(rows)} paires "
          f"(src={len(src_map)}, fr={len(fr_map)}, communs={len(common)}) "
          f"→ {out.relative_to(REPO_ROOT)}")
    return out


def main():
    p = argparse.ArgumentParser(description="Alignement paragraphique EPUB jw.org FR ↔ langue-cible.")
    p.add_argument("--lang", help="ISO (dua|ewo|bbj) ou nom (douala|ewondo|ghomala)")
    p.add_argument("--pub", help="Publication (bhs|jl|lff)")
    p.add_argument("--all", action="store_true", help="Toutes les combinaisons")
    args = p.parse_args()

    if args.all:
        for lang_dir in ISO:
            for pub in PUBS:
                align_pub(lang_dir, pub)
        return

    if not args.lang or not args.pub:
        p.error("--lang et --pub requis (sauf --all)")
    lang_dir = args.lang if args.lang in ISO else LANG_BY_ISO.get(args.lang)
    if not lang_dir:
        p.error(f"langue inconnue : {args.lang}")
    align_pub(lang_dir, args.pub)


if __name__ == "__main__":
    sys.exit(main())
