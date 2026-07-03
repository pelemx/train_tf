"""
precompute_film_cache.py — Bangun 'bridge library' FILM SEKALI, offline.

Ini implementasi ide kamu: 66 keypose + in-between hasil FILM = dataset
efektif ribuan frame, tanpa gambar manual. Setelah cache jadi, tesvid2
dengan morph_mode="film" tinggal baca PNG — model nggak pernah jalan
lagi di pipeline harian / live.

Pakai:
    # Full coverage semua pasangan unik di pool (rekomendasi, sekali jalan):
    python precompute_film_cache.py --pool phoneme_image_pool_v2.json --full

    # Atau cuma pasangan yang beneran muncul di satu render terakhir
    # (tesvid2 bisa dump timeline, lihat --from-timeline):
    python precompute_film_cache.py --pool phoneme_image_pool_v2.json \
        --from-timeline buildtemp/last_timeline.json

Fitur:
- RESUME: pasangan yang cache-nya sudah lengkap di-skip. Aman Ctrl+C.
- SIMETRI: FILM midpoint itu simetris -> (A,B) dihitung sekali,
  sequence kebalikannya disimpan gratis buat (B,A). Hemat 2x.
- ETA live di progress.

GPU GTX 1080 / CUDA 11 di Windows: pakai TF 2.10.1 (terakhir yang
support GPU native Windows) + cuDNN 8.1. Kalau TF kamu CPU-only,
tetap jalan, cuma lebih lama (jalankan semalam, sekali seumur dataset).
"""

import os
import sys
import json
import time
import argparse
import itertools

import cv2
import numpy as np

from film_interp import FilmMorpher, is_available, _load_model


def gather_unique_images(pool: dict, image_folder: str) -> list:
    names = set()
    for entry in pool.values():
        for files in entry.get("states", {}).values():
            names.update(files)
    paths = [os.path.join(image_folder, n) for n in sorted(names)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[!] {len(missing)} file di pool nggak ketemu di '{image_folder}':")
        for m in missing[:10]:
            print(f"     - {m}")
        sys.exit(1)
    return paths


def pairs_full(paths: list) -> list:
    """Semua pasangan UNORDERED (simetri FILM -> arah balik gratis)."""
    return list(itertools.combinations(paths, 2))


def pairs_from_timeline(timeline_path: str, image_folder: str) -> list:
    with open(timeline_path, "r", encoding="utf-8") as f:
        tl = json.load(f)
    seen, prev = [], None
    for e in tl:
        p = e.get("image_path")
        if not p:
            continue
        if prev is not None and p != prev:
            key = tuple(sorted((prev, p)))
            if key not in seen:
                seen.append(key)
        prev = p
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="phoneme_image_pool_v2.json")
    ap.add_argument("--image-folder", default="viseme_dataset_blended")
    ap.add_argument("--cache-dir", default="buildtemp/film_cache")
    ap.add_argument("--levels", type=int, default=3,
                    help="3 -> 7 in-between per pasangan (default)")
    ap.add_argument("--full", action="store_true",
                    help="semua pasangan unik di pool")
    ap.add_argument("--from-timeline", default=None,
                    help="path JSON timeline (entry punya 'image_path')")
    args = ap.parse_args()

    if not is_available():
        print("[!] tensorflow / tensorflow_hub belum terinstall.")
        print("    GTX 1080 + CUDA 11 (Windows): pip install tensorflow==2.10.1 tensorflow_hub")
        sys.exit(1)

    # info device
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[*] TensorFlow {tf.__version__} | GPU: "
          f"{gpus[0].name if gpus else 'NONE (CPU mode — lebih lambat tapi jalan)'}")

    with open(args.pool, "r", encoding="utf-8") as f:
        pool = json.load(f)

    if args.from_timeline:
        pairs = pairs_from_timeline(args.from_timeline, args.image_folder)
        print(f"[*] Mode: pasangan dari timeline -> {len(pairs)} pasangan")
    else:
        paths = gather_unique_images(pool, args.image_folder)
        pairs = pairs_full(paths)
        print(f"[*] Mode: FULL coverage -> {len(paths)} image unik, "
              f"{len(pairs)} pasangan unordered")

    fm = FilmMorpher(cache_dir=args.cache_dir, levels=args.levels)
    calls_per_pair = (2 ** args.levels) - 1   # midpoint calls per pasangan

    # cek resume: pasangan yang cache-nya (dua arah) sudah lengkap
    def cached(a, b):
        for key in ((a, b), (b, a)):
            ph = fm._pair_hash(key)
            if all(os.path.exists(p) for p in fm._disk_paths(ph)):
                return True
        return False

    todo = [(a, b) for (a, b) in pairs if not cached(a, b)]
    print(f"[*] {len(pairs) - len(todo)} pasangan sudah di cache, "
          f"{len(todo)} tersisa ({len(todo) * calls_per_pair} FILM calls)")
    if not todo:
        print("[OK] Cache sudah lengkap. Selesai.")
        return

    _load_model()  # download/load sekali di awal biar ETA akurat

    img_cache = {}
    def get_img(p):
        if p not in img_cache:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(p)
            img_cache[p] = np.ascontiguousarray(im)
        return img_cache[p]

    t0 = time.time()
    for i, (a, b) in enumerate(todo):
        seq = fm.get_sequence(get_img(a), get_img(b), (a, b))
        # arah balik = sequence dibalik, disimpan tanpa FILM call
        ph_rev = fm._pair_hash((b, a))
        rev_paths = fm._disk_paths(ph_rev)
        if not all(os.path.exists(p) for p in rev_paths):
            fm._save_to_disk(ph_rev, list(reversed(seq)))

        done = i + 1
        elapsed = time.time() - t0
        rate = elapsed / done
        eta = rate * (len(todo) - done)
        print(f"    [{done}/{len(todo)}] "
              f"{os.path.basename(a)} <-> {os.path.basename(b)} | "
              f"{rate:.1f}s/pair | ETA {eta/60:.1f} min")

    total_frames = len(pairs) * ((2 ** args.levels) + 1)
    print(f"\n[OK] Bridge library selesai: {len(pairs)} pasangan, "
          f"~{total_frames} frame di {args.cache_dir}")
    print("     Set CONFIG['morph_mode'] = 'film' di tesvid2.py — "
          "render selanjutnya cuma baca PNG, zero model cost.")


if __name__ == "__main__":
    main()