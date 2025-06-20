import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

GENRES = ['hiphop', 'classical', 'jazz', 'rock']
AUDIO_DIR = 'finetune_data3'
FINETUNE3_OUTPUT_DIR = 'spectrograms_finetune3'
IMG_SIZE = (2.24, 2.24)
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  

def ensure_dirs():
    for genre in GENRES:
        os.makedirs(os.path.join(FINETUNE3_OUTPUT_DIR, genre), exist_ok=True)

def generate_spectrogram(segment, sr, output_path):
    S = librosa.feature.melspectrogram(y=segment, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if np.max(S_dB) < -80:
        print(f"⚠️ Pominięto pusty/cichy segment")
        return

    plt.figure(figsize=IMG_SIZE)
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_all():
    ensure_dirs()
    for genre in GENRES:
        genre_path = os.path.join(AUDIO_DIR, genre)
        if not os.path.exists(genre_path):
            print(f"⚠️ Pominięto: {genre_path} nie istnieje")
            continue

        for file in os.listdir(genre_path):
            if not file.lower().endswith('.wav'):
                continue

            filepath = os.path.join(genre_path, file)
            base_name = os.path.splitext(file)[0]

            try:
                y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"❌ Błąd wczytywania {filepath}: {e}")
                continue

            total_segments = int(len(y) / (SEGMENT_DURATION * sr))
            for i in range(total_segments):
                start = i * SEGMENT_DURATION * sr
                end = start + SEGMENT_DURATION * sr
                segment = y[int(start):int(end)]

                out_name = f"{base_name}_ft2_{i}.png"
                out_path = os.path.join(FINETUNE3_OUTPUT_DIR, genre, out_name)
                if os.path.exists(out_path):
                    continue 
                generate_spectrogram(segment, sr, out_path)
                print(f"✅ zapisano: {genre}/{out_name}")

if __name__ == '__main__':
    process_all()