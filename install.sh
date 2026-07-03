#!/bin/bash
# run_film.sh

mkdir -p bridge_build
cd bridge_build || exit

echo "[*] Downloading dataset from HuggingFace..."
wget -qO viseme_dataset_blended.zip "https://huggingface.co/buckets/golekpelem/jupcore/resolve/viseme_dataset_blended.zip?download=true"
unzip -qo viseme_dataset_blended.zip

echo "[*] Initializing virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[*] Installing TensorFlow and FILM (Background process)..."
pip install --no-cache-dir "tensorflow[and-cuda]" tensorflow_hub opencv-python-headless numpy
pip install --no-cache-dir git+https://github.com/google-research/frame-interpolation

echo "[*] Executing FILM cache precomputation..."
# Ensure precompute_film_cache.py and phoneme_image_pool_v2.json are uploaded here first
python precompute_film_cache.py --pool phoneme_image_pool_v2.json --full --engine film

echo "[*] Compressing output..."
tar czf film_cache.tgz buildtemp/film_cache
echo "[*] Process complete. Target ready for download."
