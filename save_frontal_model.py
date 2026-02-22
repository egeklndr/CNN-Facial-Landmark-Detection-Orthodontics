import os
import json
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("="*60)
print("FRONTAL MODEL TRAINING AND RECORDING")
print("="*60)

# 1. DATA UPLOAD

print("\n Loading frontal data...")

images = []
landmarks = []

for i in range(1, 99):
    json_file = f"clabel/C{i}.json"
    
    if not os.path.exists(json_file):
        continue
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        continue
    
    # Image path - in the cvisual folder
    img_filename = data.get('imagePath', '')
    img_path = f"cvisual/{img_filename}"
    
    if not os.path.exists(img_path):
        continue
    
    # Upload image
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    
    # Landmarks
    points = []
    if 'shapes' in data:
        for shape in data['shapes']:
            if 'points' in shape and len(shape['points']) > 0:
                point = shape['points'][0]
                x = point[0] / img_w
                y = point[1] / img_h
                points.extend([x, y])
    
    if len(points) == 44:  # 22 landmarks
        images.append(img)
        landmarks.append(points)

X = np.array(images)
y = np.array(landmarks)

print(f" Uploaded: {len(X)} images")
print(f"   Shape: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print(" DATA COULD NOT BE LOADED!")
    exit(1)

# 2. DATA SPLIT

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"\n Split:")
print(f"   Train: {len(X_train)}")
print(f"   Val: {len(X_val)}")
print(f"   Test: {len(X_test)}")

# 3. CREATE A MODEL

print("\n The model is being created...")

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3,3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(44, activation='sigmoid')
], name='frontal_Custom_CNN')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae']
)

print(f" Model: {model.count_params():,} parameters")

# 4. TRAINING 

print("\n Training begins...")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# 5. TEST

print("\n Test evaluation...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"   Test Loss: {test_loss:.6f}")
print(f"   Test MAE: {test_mae:.6f}")

# 6. SAVE

os.makedirs('results/models', exist_ok=True)

model_path = 'results/models/frontal_Custom_CNN_TRAINED.keras'
model.save(model_path)

size_mb = os.path.getsize(model_path) / (1024 * 1024)

print(f"\n FRONTAL MODEL SAVED:")
print(f"   File: {model_path}")
print(f"   Size: {size_mb:.2f} MB")

# Info 
info_text = f"""
MODEL PERFORMANCE (FRONTAL)
========================================
Test Loss (MSE): {test_loss:.6f}
Test MAE: {test_mae:.6f}
Number of landmarks: 22
Test set size: {len(X_test)} images
"""

with open('results/models/frontal_model_info.txt', 'w') as f:
    f.write(info_text)

print(info_text)

print("\n COMPLETED!")