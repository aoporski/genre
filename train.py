import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, ResNet50
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_DIR = 'spectrograms'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 4  

def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def build_mobilenet():
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def build_resnet():
    base = ResNet50(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def get_data():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.2
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

    plt.savefig('training_plots.png')
    plt.show()


def evaluate_model(model, val_gen):
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

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
    plt.savefig('confusion_matrix.png')
    plt.show()


def train_model(model_name):
    train_gen, val_gen = get_data()
    os.makedirs('models', exist_ok=True)

    if model_name == 'cnn':
        model = build_cnn()
    elif model_name == 'mobilenet':
        model = build_mobilenet()
    elif model_name == 'resnet':
        model = build_resnet()
    else:
        raise ValueError("❌ Model must be one of: cnn, mobilenet, resnet")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(f'models/{model_name}_best_3v2_fixed.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        #  EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # dla modelu 1 i 2
        EarlyStopping(monitor='val_accuracy',  patience=5, restore_best_weights=True,mode='max')]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    plot_history(history)
    evaluate_model(model, val_gen)
    print(f"✅ Zapisano model jako models/{model_name}_best_3v2_fixed.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="cnn | mobilenet | resnet")
    args = parser.parse_args()
    train_model(args.model)