import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# CONFIGURATION

MODELS_DIR = 'results/models'
os.makedirs(MODELS_DIR, exist_ok=True)

STATS_CSV = 'results/statistical_validation.csv'

# MODEL ARCHITECTURE

def create_cnn_model(num_landmarks, name):
    """Create CNN model"""
    model = keras.Sequential(name=name)
    
    model.add(layers.Input(shape=(128, 128, 3)))
    
    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Block 4
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_landmarks * 2, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# CREATE AND SAVE ARCHITECTURE MODELS

def create_architecture_models():
    """Create model architectures (without training)"""
    
    print("="*60)
    print("CREATING MODEL ARCHITECTURES")
    print("="*60)
    
    models_created = []
    
    # Frontal model (22 landmarks)
    print("\n Creating FRONTAL model...")
    frontal_model = create_cnn_model(22, "frontal_Custom_CNN")
    frontal_path = os.path.join(MODELS_DIR, 'frontal_Custom_CNN_architecture.keras')
    frontal_model.save(frontal_path)
    
    size_mb = os.path.getsize(frontal_path) / (1024 * 1024)
    print(f"   Saved: {frontal_path}")
    print(f"   Parameters: {frontal_model.count_params():,}")
    print(f"   Size: {size_mb:.2f} MB")
    models_created.append(('FRONTAL', frontal_path, size_mb))
    
    # Profile model (15 landmarks)
    print("\n Creating PROFILE model...")
    profile_model = create_cnn_model(15, "profile_Custom_CNN")
    profile_path = os.path.join(MODELS_DIR, 'profile_Custom_CNN_architecture.keras')
    profile_model.save(profile_path)
    
    size_mb = os.path.getsize(profile_path) / (1024 * 1024)
    print(f"   Saved: {profile_path}")
    print(f"   Parameters: {profile_model.count_params():,}")
    print(f"   Size: {size_mb:.2f} MB")
    models_created.append(('PROFILE', profile_path, size_mb))
    
    # Create README
    readme_path = os.path.join(MODELS_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# CNN Model Architectures\n\n")
        f.write("These are the architecture files for the Custom CNN models.\n\n")
        f.write("## Models\n\n")
        f.write("- `frontal_Custom_CNN_architecture.keras` - Frontal view model (22 landmarks)\n")
        f.write("- `profile_Custom_CNN_architecture.keras` - Profile view model (15 landmarks)\n\n")
        f.write("## Architecture Details\n\n")
        f.write("- Input: 128x128x3 RGB images\n")
        f.write("- 4 convolutional blocks (32→64→128→256 filters)\n")
        f.write("- Kernel size: (3,3)\n")
        f.write("- Pooling: (2,2) MaxPooling\n")
        f.write("- Dense layers: 512 units + Dropout(0.5)\n")
        f.write("- Optimizer: Adam (LR=0.0001)\n")
        f.write("- Loss: MSE\n\n")
        f.write("## Performance (from paper)\n\n")
        
        # Read stats
        if os.path.exists(STATS_CSV):
            stats = pd.read_csv(STATS_CSV)
            f.write("```\n")
            f.write(stats.to_string(index=False))
            f.write("\n```\n")
        
        f.write("\n\n## Usage\n\n")
        f.write("```python\n")
        f.write("from tensorflow import keras\n")
        f.write("model = keras.models.load_model('frontal_Custom_CNN_architecture.keras')\n")
        f.write("```\n")
    
    print(f"\n✅ README created: {readme_path}")
    
    return models_created

# MAIN

if __name__ == "__main__":
    print("="*60)
    print("CNN MODEL ARCHITECTURE EXPORT")
    print("="*60)
    
    models = create_architecture_models()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, path, size in models:
        print(f" {name}: {size:.2f} MB")
    
    print("\n" + "="*60)
    print("NOTES")
    print("="*60)
    print(" These are UNTRAINED architecture models")
    print(" For trained weights, use the models from notebook")
    print(" Architecture is identical to paper")
    print(" Can be used for reproducibility")
    
    print("\n ALL DONE!")