#!/usr/bin/env python3
"""
Aspire les publications jw.org pour nos 5 langues cibles.

Ce script tape sur l'API publique app.jw-cdn.org/apis/pub-media pour
récupérer, pour chaque couple (langue, publication) déjà validé lors du
recensement des sources, la liste des fichiers média disponibles
(audio MP3, texte EPUB, texte PDF) puis les télécharge dans
data/{langue}/jw_org/.

La correspondance langue ISO ↔ code interne jw.org et la liste des
publications réellement disponibles ont été établies dans
../data/SOURCES.md à partir d'un balayage exhaustif de l'API.

Usage:
    python scripts/scrape_jw.py                # toutes les langues
    python scripts/scrape_jw.py --lang ewo     # seulement Ewondo
    python scripts/scrape_jw.py --french       # + la version FR (pour alignement)
    python scripts/scrape_jw.py --dry-run      # liste sans télécharger

Sortie:
    data/{lang}/jw_org/{audio,text}/…    fichiers média (gitignorés)
    data/jw_org_manifest.json            manifeste consolidé
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
API_URL = "https://app.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS"
UA = "Mozilla/5.0 (MBOA-VOICE research corpus bootstrap; contact mbia1378@gmail.com)"

# Code langue interne jw.org ↔ dossier local ISO
LANGUAGES = {
    "ewondo":   {"jw": "EWN", "iso": "ewo"},
    "douala":   {"jw": "DA",  "iso": "dua"},
    "fulfulde": {"jw": "FD",  "iso": "fub"},
    "ghomala":  {"jw": "GHM", "iso": "bbj"},
    # Fe'efe'e (FFE) : rien trouvé via l'API pub-media lors du recon,
    # laissé pour trace au cas où le catalogue serait mis à jour.
    "feefee":   {"jw": "FFE", "iso": "fmp"},
}

# Publications à récupérer par langue (source: data/SOURCES.md, recon
# validé le 15/09/2026). Chaque tuple = (dossier, code_pub, formats).
JOBS: list[tuple[str, str, list[str]]] = [
    ("ewondo",   "lff",  ["EPUB", "PDF"]),      # Vivez pour toujours
    ("ewondo",   "sjjm", ["MP3"]),              # Recueil de chants (159 pistes)
    ("ewondo",   "jcr",  ["MP3"]),              # 1 piste
    ("douala",   "jl",   ["MP3", "EPUB", "PDF"]),  # Jésus (31 chapitres)
    ("douala",   "bhs",  ["EPUB", "PDF"]),      # Bible enseigne
    ("douala",   "sjjm", ["MP3"]),              # Recueil de chants (163 pistes)
    ("douala",   "lff",  ["PDF"]),
    ("fulfulde", "sjjm", ["MP3"]),              # Recueil de chants (151 pistes)
    ("ghomala",  "jl",   ["MP3", "EPUB", "PDF"]),  # Jésus (31 chapitres)
    ("ghomala",  "lff",  ["PDF"]),
    # feefee : rien à date
]


def api_get(lang_jw: str, pub: str, fmt: str) -> dict | None:
    url = f"{API_URL}?langwritten={lang_jw}&pub={pub}&fileformat={fmt}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def iter_tracks(payload: dict):
    """Aplati la structure imbriquée { files: {LANG: {FMT: [tracks]}} }."""
    for _lang, formats in (payload.get("files") or {}).items():
        for _fmt, tracks in formats.items():
            for t in tracks:
                yield t


def download(url: str, dest: pathlib.Path, expected_md5: str | None) -> str:
    if dest.exists():
        if expected_md5 and hashlib.md5(dest.read_bytes()).hexdigest() == expected_md5:
            return "cached"
        # taille non nulle, on suppose OK
        if not expected_md5 and dest.stat().st_size > 0:
            return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    return "downloaded"


def process(data_dir: str, lang_jw: str, pub: str, fmts: list[str],
            throttle: float, dry_run: bool) -> list[dict]:
    entries = []
    for fmt in fmts:
        payload = api_get(lang_jw, pub, fmt)
        if not payload:
            print(f"  [skip] {lang_jw}/{pub}/{fmt}: 404")
            continue
        for track in iter_tracks(payload):
            file_info = track.get("file") or {}
            url = file_info.get("url")
            if not url:
                continue
            ext = url.rsplit(".", 1)[-1].lower()
            subdir = "audio" if ext in {"mp3", "aac", "m4a"} else "text"
            track_num = track.get("track") or 0
            fname = f"{pub}_{track_num:02d}.{ext}" if subdir == "audio" else f"{pub}.{ext}"
            dest = REPO_ROOT / "data" / data_dir / "jw_org" / subdir / fname
            entry = {
                "lang_jw": lang_jw,
                "pub": pub,
                "fmt": fmt,
                "title": track.get("title"),
                "track": track_num,
                "duration": track.get("duration"),
                "url": url,
                "path": str(dest.relative_to(REPO_ROOT)),
                "size": file_info.get("filesize"),
            }
            if dry_run:
                entry["status"] = "dry-run"
            else:
                entry["status"] = download(url, dest, file_info.get("checksum"))
                time.sleep(throttle)
            entries.append(entry)
            print(f"  [{entry['status']:>10s}] {dest.name} ({entry['size']} B)")
    return entries


def main():
    p = argparse.ArgumentParser(description="Aspiration corpus jw.org multilingue.")
    p.add_argument("--lang", default="all", help="ISO 639-3 (ewo, dua, fub, bbj, fmp) ou 'all'")
    p.add_argument("--french", action="store_true", help="ajouter la version FR pour alignement")
    p.add_argument("--throttle", type=float, default=0.8, help="pause entre téléchargements (s)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = []
    for data_dir, pub, fmts in JOBS:
        iso = LANGUAGES[data_dir]["iso"]
        if args.lang not in ("all", iso):
            continue
        print(f"\n=== {data_dir} ({iso}) / pub={pub} ===")
        manifest.extend(process(data_dir, LANGUAGES[data_dir]["jw"], pub, fmts,
                                args.throttle, args.dry_run))

    if args.french:
        # Récupère la version FR de chaque publication rencontrée
        seen: set[tuple[str, str]] = set()
        for _dd, pub, fmts in JOBS:
            for fmt in fmts:
                if (pub, fmt) in seen:
                    continue
                seen.add((pub, fmt))
                print(f"\n=== _french / pub={pub} / fmt={fmt} ===")
                manifest.extend(process("_french", "F", pub, [fmt],
                                        args.throttle, args.dry_run))

    if not args.dry_run:
        out = REPO_ROOT / "data" / "jw_org_manifest.json"
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nManifeste : {out}")
    print(f"Total: {len(manifest)} pistes traitées.")


if __name__ == "__main__":
    sys.exit(main())
