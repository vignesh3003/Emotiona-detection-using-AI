import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- Step 1: Preprocessing with Data Augmentation ---
train_dir = '/Users/vigneshraj/emotion_recognition/train'  # Replace with your training directory
test_dir = '/Users/vigneshraj/emotion_recognition/test'  # Replace with your test directory

# Image preprocessing with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for test data, only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # EfficientNet default input size
    color_mode='rgb',  # EfficientNet expects RGB
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical'
)

# --- Step 2: Build the EfficientNet Model ---
# Load the EfficientNetB0 model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's layers
base_model.trainable = False

# Add custom classification layers on top of EfficientNet
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Replaces Flatten(), works well with EfficientNet
    Dropout(0.5),  # Helps prevent overfitting
    Dense(128, activation='relu'),  # Custom fully connected layer
    Dropout(0.5),  # More dropout to prevent overfitting
    Dense(7, activation='softmax')  # Output layer for 7 emotion classes
])

# --- Step 3: Compile the Model ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# --- Step 4: Set Callbacks ---
# Learning rate scheduler to reduce learning rate on plateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Step 5: Train the Model ---
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,  # Adjust based on early stopping and performance
    steps_per_epoch=train_generator.samples // 32,
    validation_steps=test_generator.samples // 32,
    callbacks=[lr_scheduler, early_stopping]
)

# --- Step 6: Fine-tune the Model ---
# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Compile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model for a few more epochs
fine_tune_history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # Usually fine-tuning requires fewer epochs
    steps_per_epoch=train_generator.samples // 32,
    validation_steps=test_generator.samples // 32,
    callbacks=[lr_scheduler, early_stopping]
)

# --- Step 7: Evaluate the Model ---
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the final model
model.save('/path/to/save/emotion_detection_efficientnet.h5')
