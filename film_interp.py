"""
film_interp.py — Google FILM interpolation, dipakai dengan cara HEMAT.

Trik utamanya: JANGAN jalanin FILM per-frame video. Pool image kamu
fixed (~30 file), jadi pasangan transisi unik cuma ratusan. FILM
dijalanin SEKALI per pasangan unik, hasilnya (7 in-between frames)
di-cache ke disk. Generate video berikutnya = tinggal baca PNG cache,
FILM nggak pernah jalan lagi. Model-quality interpolation, zero cost
di pipeline harian.

Install (sekali):
    pip install tensorflow tensorflow_hub
Model auto-download dari TF Hub pas pertama dipanggil (~50MB, disimpan
di ~/.cache). Kalau TF nggak ada, is_available() -> False dan caller
(tesvid2.py) otomatis fallback ke FlowMorpher.

Pakai:
    fm = FilmMorpher(cache_dir="buildtemp/film_cache", levels=3)
    frame = fm.morph(imgA, imgB, alpha, pair_key=(pathA, pathB))
"""

import os
import hashlib
import cv2
import numpy as np

_MODEL = None
_TF = None


def is_available() -> bool:
    try:
        import tensorflow  # noqa
        import tensorflow_hub  # noqa
        return True
    except ImportError:
        return False


def _load_model():
    global _MODEL, _TF
    if _MODEL is None:
        import tensorflow as tf
        import tensorflow_hub as hub
        _TF = tf
        print("[*] Loading FILM model from TF Hub (first run downloads ~50MB)...")
        _MODEL = hub.load("https://tfhub.dev/google/film/1")
    return _MODEL


def _film_midpoint(img1_bgr, img2_bgr):
    """Satu frame tengah (t=0.5) antara dua image BGR uint8."""
    model = _load_model()
    tf = _TF
    a = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    b = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = {
        "x0": tf.expand_dims(a, 0),
        "x1": tf.expand_dims(b, 0),
        "time": tf.constant([[0.5]], dtype=tf.float32),
    }
    mid = model(inp)["image"][0].numpy()
    mid = np.clip(mid * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mid, cv2.COLOR_RGB2BGR)


class FilmMorpher:
    """
    Drop-in pengganti FlowMorpher.morph() dengan signature sama.

    levels=3 -> rekursif midpoint 3x -> 7 in-between + 2 endpoint
    = 9 titik alpha. morph(alpha) nge-snap ke titik terdekat; setelah
    RIFE 30->60fps hasilnya udah mulus banget.
    """

    def __init__(self, cache_dir="buildtemp/film_cache", levels=3,
                 allow_compute=None):
        """allow_compute: True = boleh hitung pair baru (precompute mode,
        engine apapun). None/auto = hitung cuma kalau TF ada. False =
        cache-only (runtime tanpa TF)."""
        self.cache_dir = cache_dir
        self.levels = levels
        self.n_points = (2 ** levels) + 1   # termasuk endpoint
        self.allow_compute = allow_compute
        os.makedirs(cache_dir, exist_ok=True)
        self._mem = {}   # pair_hash -> list[np.ndarray] panjang n_points

    # ── cache helpers ──────────────────────────────────
    def _pair_hash(self, pair_key):
        # PENTING: hash pakai BASENAME, bukan full path. Full path beda
        # antar OS (slash vs backslash) & antar folder -> cache yang
        # dibangun di server Linux nggak akan kebaca di Windows.
        # Basename bikin cache 100% portable (scp/copy antar mesin).
        parts = [os.path.basename(str(k)) for k in pair_key]
        s = "|".join(parts)
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _disk_paths(self, ph):
        return [os.path.join(self.cache_dir, f"{ph}_{i:02d}.png")
                for i in range(self.n_points)]

    def _load_from_disk(self, ph):
        paths = self._disk_paths(ph)
        if all(os.path.exists(p) for p in paths):
            return [cv2.imread(p) for p in paths]
        return None

    def _save_to_disk(self, ph, frames):
        for p, f in zip(self._disk_paths(ph), frames):
            cv2.imwrite(p, f)

    # ── core ───────────────────────────────────────────
    def _build_sequence(self, img1, img2):
        """Rekursif midpoint: [A, ..., B] dengan (2^levels)+1 titik."""
        seq = [img1, img2]
        for _ in range(self.levels):
            nxt = [seq[0]]
            for i in range(len(seq) - 1):
                nxt.append(_film_midpoint(seq[i], seq[i + 1]))
                nxt.append(seq[i + 1])
            seq = nxt
        return seq

    def get_sequence(self, img1, img2, pair_key):
        ph = self._pair_hash(pair_key)
        if ph in self._mem:
            return self._mem[ph]
        frames = self._load_from_disk(ph)
        if frames is None:
            # Cache miss. Boleh hitung kalau: allow_compute=True
            # (precompute mode, engine bebas — FILM atau RIFE via patch),
            # atau auto (None) dan TF tersedia. Kalau nggak -> None,
            # caller fallback ke flow morph. Runtime TIDAK butuh TF
            # selama cache lengkap.
            can = self.allow_compute if self.allow_compute is not None \
                else is_available()
            if not can:
                return None
            frames = self._build_sequence(img1, img2)
            self._save_to_disk(ph, frames)
        self._mem[ph] = frames
        return frames

    def morph(self, img1, img2, alpha, pair_key=None):
        if alpha <= 0.0:
            return img1
        if alpha >= 1.0:
            return img2
        if pair_key is None:
            pair_key = ("anon", id(img1), id(img2))
        seq = self.get_sequence(img1, img2, pair_key)
        if seq is None:
            return None   # caller (tesvid2) fallback ke flow morph
        # smoothstep easing, lalu snap ke titik precomputed terdekat
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        idx = int(round(eased * (len(seq) - 1)))
        return seq[idx]


def precompute_pairs(timeline, get_img, morpher):
    """
    Opsional: warm-up cache SEBELUM render, biar progress kelihatan.
    timeline = list entry dengan 'image_path'; get_img = loader.
    """
    pairs = []
    prev = None
    for e in timeline:
        p = e.get("image_path")
        if prev is not None and p != prev:
            key = (prev, p)
            if key not in pairs:
                pairs.append(key)
        prev = p
    print(f"[*] FILM precompute: {len(pairs)} unique transitions...")
    for i, (a, b) in enumerate(pairs):
        morpher.get_sequence(get_img(a), get_img(b), (a, b))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(pairs)}")
    print("[OK] FILM cache warm.")
