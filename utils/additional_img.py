import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# KONFIGURACJA
GENRES = ['rock', 'jazz', 'classical', 'hiphop']
AUDIO_DIR = 'additional_data'  
OUTPUT_DIR = 'spectrograms'    
IMG_SIZE = (2.24, 2.24)
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for genre in GENRES:
        os.makedirs(os.path.join(OUTPUT_DIR, genre), exist_ok=True)

def generate_spectrogram(segment, sr, output_path):
    S = librosa.feature.melspectrogram(y=segment, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=IMG_SIZE)
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert_all():
    ensure_dirs()
    for genre in GENRES:
        genre_dir = os.path.join(AUDIO_DIR, genre)
        output_genre_dir = os.path.join(OUTPUT_DIR, genre)

        if not os.path.exists(genre_dir):
            print(f"⚠️ Pominięto brakujący katalog: {genre_dir}")
            continue

        for file in os.listdir(genre_dir):
            if not file.lower().endswith('.wav'):
                continue

            filepath = os.path.join(genre_dir, file)
            base_name = os.path.splitext(file)[0]

            try:
                y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"❌ Błąd wczytywania {filepath}: {e}")
                continue

            for i in range(10):
                start = i * SEGMENT_DURATION * sr
                end = start + SEGMENT_DURATION * sr
                segment = y[int(start):int(end)]

                if len(segment) < SEGMENT_DURATION * sr:
                    continue

                out_name = f"{base_name}_{i}.png"
                out_path = os.path.join(output_genre_dir, out_name)

                try:
                    generate_spectrogram(segment, sr, out_path)
                    print(f"✅ {genre}: {out_name}")
                except Exception as e:
                    print(f"⚠️ Błąd podczas generowania spektrogramu: {out_name} — {e}")

if __name__ == '__main__':
    convert_all()