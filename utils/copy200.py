import os
import shutil
from collections import defaultdict
import random

SOURCE_DIR = 'spectrograms_finetune3'
TARGET_DIR = 'balanced_test'
SAMPLES_PER_CLASS = 200

os.makedirs(TARGET_DIR, exist_ok=True)

for genre in os.listdir(SOURCE_DIR):
    genre_path = os.path.join(SOURCE_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    target_genre_path = os.path.join(TARGET_DIR, genre)
    os.makedirs(target_genre_path, exist_ok=True)

    files = [f for f in os.listdir(genre_path) if f.lower().endswith('.png')]
    random.shuffle(files)
    selected = files[:SAMPLES_PER_CLASS]

    for f in selected:
        shutil.copy(os.path.join(genre_path, f), os.path.join(target_genre_path, f))

    print(f"✅ Skopiowano {len(selected)} plików z klasy: {genre}")