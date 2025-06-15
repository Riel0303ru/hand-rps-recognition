import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
import os
import pickle
from collections import Counter

# 1. Path dataset
base_dir = "organized_dataset"

# 2. Augmentasi & preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 3. Class weight untuk handle ketidakseimbangan data
counter = Counter(train_generator.classes)
total = sum(counter.values())
class_weight = {i: total / (len(counter) * count) for i, count in counter.items()}

# 4. Transfer Learning dengan MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Training dengan EarlyStopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[
        TqdmCallback(verbose=1),
        early_stop
    ]
)

# 6. Simpan model dan riwayat training
os.makedirs("model", exist_ok=True)
model.save("model/rps_model.h5")

with open("model/history.pkl", "wb") as f:
    pickle.dump(history.history, f)
