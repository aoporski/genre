import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
from io import BytesIO

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
IMG_SIZE = (224, 224)
GENRES = ['classical', 'hiphop', 'jazz', 'rock']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('models/mobilenet_finetuned_final.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_audio(y, sr, segment_duration=3, stride=6):
    segment_len = segment_duration * sr
    stride_len = stride * sr
    return [y[i:i+segment_len] for i in range(0, len(y) - segment_len + 1, stride_len)]

def audio_segments_to_images(y, sr):
    segments = split_audio(y, sr)
    images = []

    for segment in segments:
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, cmap='magma')
        ax.axis('off')
        fig.tight_layout(pad=0)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert('RGB').resize(IMG_SIZE)
        images.append(np.array(img) / 255.0)

    return np.array(images)

def predict_genre_segments(y, sr):
    images = audio_segments_to_images(y, sr)
    if len(images) == 0:
        return "unknown", 0.0

    preds = model.predict(images)
    avg_pred = np.mean(preds, axis=0)
    genre_idx = np.argmax(avg_pred)
    return GENRES[genre_idx], float(avg_pred[genre_idx])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)

            y, sr = librosa.load(audio_path, sr=22050)
            genre, confidence = predict_genre_segments(y, sr)

            confidence_val = confidence * 100
            return render_template(
                'index.html',
                genre=genre,
                confidence_display=f"{confidence_val:.2f}%",
                confidence=confidence_val, 
                filename=filename
            )

    return render_template('index.html', genre=None)

if __name__ == '__main__':
    app.run(debug=True)