import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import os

# ================================
# ✅ DATASET PATH
# ================================
train_dir = r'C:\project\dataset\train'

if not os.path.exists(train_dir):
    print(f"❌ Dataset not found at: {train_dir}")
    exit()

print("✅ Dataset found!")

# ================================
# ✅ CHECK IMAGE COUNTS
# ================================
print("\n📊 Image counts per class:")
total = 0
for folder in sorted(os.listdir(train_dir)):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'))
        ])
        print(f"   {folder}: {count} images")
        total += count
print(f"   TOTAL: {total} images\n")

# ================================
# ✅ DATA AUGMENTATION
# ResNet50 needs 224x224 images
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# ================================
# ✅ LOAD DATA - 224x224 for ResNet50
# ================================
train = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # ✅ ResNet50 needs 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✅ Classes found ({train.num_classes}):")
for name, idx in sorted(train.class_indices.items()):
    print(f"   {idx}: {name}")

print(f"\n✅ Training images:   {train.samples}")
print(f"✅ Validation images: {val.samples}")

# ================================
# ✅ SAVE CLASS LABELS
# ================================
os.makedirs(r'C:\project\model', exist_ok=True)
with open(r'C:\project\model\classes.json', 'w') as f:
    json.dump(train.class_indices, f, indent=2)
print("\n✅ classes.json saved!")

# ================================
# ✅ RESNET50 MODEL
# ================================
print("\n🔬 Building ResNet50 Transfer Learning Model...")

# Load ResNet50 pretrained on ImageNet (1.2M images, 1000 classes)
base_model = ResNet50(
    weights='imagenet',        # ✅ Pre-trained weights
    include_top=False,         # ✅ Remove last classification layer
    input_shape=(224, 224, 3)
)

# ✅ Phase 1: Freeze all ResNet50 layers
# Only train our custom top layers first
base_model.trainable = False
print(f"✅ ResNet50 loaded - {len(base_model.layers)} layers frozen")

# ================================
# ✅ ADD CUSTOM TOP LAYERS
# ================================
model = tf.keras.models.Sequential([
    base_model,

    # Global average pooling instead of flatten
    tf.keras.layers.GlobalAveragePooling2D(),

    # Dense layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    # Output layer
    tf.keras.layers.Dense(train.num_classes, activation='softmax')
])

# ================================
# ✅ PHASE 1 - COMPILE & TRAIN TOP LAYERS
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks_phase1 = [
    ModelCheckpoint(
        r'C:\project\model\model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

print("\n🚀 Phase 1: Training top layers (ResNet50 frozen)...")
print("=" * 50)

history1 = model.fit(
    train,
    validation_data=val,
    epochs=15,
    callbacks=callbacks_phase1,
    verbose=1
)

print(f"\n✅ Phase 1 complete!")
print(f"   Training accuracy:   {history1.history['accuracy'][-1]*100:.2f}%")
print(f"   Validation accuracy: {history1.history['val_accuracy'][-1]*100:.2f}%")

# ================================
# ✅ PHASE 2 - FINE TUNING
# Unfreeze last 30 layers of ResNet50
# ================================
print("\n🔬 Phase 2: Fine-tuning ResNet50 last 30 layers...")

base_model.trainable = True

# Freeze all layers EXCEPT last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

unfrozen = sum(1 for l in base_model.layers if l.trainable)
print(f"✅ Unfrozen {unfrozen} layers for fine-tuning")

# Use LOWER learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    ModelCheckpoint(
        r'C:\project\model\model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

print("\n🚀 Phase 2: Fine-tuning started...")
print("=" * 50)

history2 = model.fit(
    train,
    validation_data=val,
    epochs=25,
    callbacks=callbacks_phase2,
    verbose=1
)

# ================================
# ✅ SAVE FINAL MODEL
# ================================
model.save(r'C:\project\model\model.h5')

print("\n" + "=" * 50)
print("🎉 ResNet50 Training Complete!")
print("=" * 50)
print(f"📊 Phase 1 best accuracy: {max(history1.history['val_accuracy'])*100:.2f}%")
print(f"📊 Phase 2 best accuracy: {max(history2.history['val_accuracy'])*100:.2f}%")
print(f"✅ Model saved: C:\\project\\model\\model.h5")
print("\n📋 Final class labels:")
for name, idx in sorted(train.class_indices.items()):
    print(f"   {idx}: {name}")