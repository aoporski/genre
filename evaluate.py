import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = 'models/mobilenet_finetuned_final.h5' 
TEST_DIR = 'balanced_test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 1

model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

print("🧪 Raport z testowania:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Oranges')
plt.title("Macierz pomyłek – zbiór testowy")
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
plt.colorbar()
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.tight_layout()
plt.savefig('confusion_matrix_test.png')
plt.show()