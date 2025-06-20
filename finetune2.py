
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = 'spectrograms_finetune2'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-6
MODEL_INPUT_PATH = 'models/mobilenet_finetuned_fixed.h5'
MODEL_OUTPUT_PATH = 'models/mobilenet_finetuned_final.h5'

def get_data():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train, val

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.savefig('final_training_plots.png')
    plt.show()

def evaluate_model(model, val_gen):
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open("final_classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    plt.show()

def finetune_model():
    train_gen, val_gen = get_data()

    model = load_model(MODEL_INPUT_PATH)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-40:]:
        layer.trainable = True

    print(f"Trainable layers: {sum([layer.trainable for layer in model.layers])}/{len(model.layers)}")

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(MODEL_OUTPUT_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, mode='max', min_lr=1e-7)
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    plot_history(history)
    evaluate_model(model, val_gen)
    print(f"✅ Zapisano model jako {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    finetune_model()
