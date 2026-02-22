# CNN-Based Facial Landmark Detection for Orthodontic Cephalometry

Custom Convolutional Neural Network for automated 2D facial soft tissue landmark detection in orthodontic applications.

## Performance

- **Mean Error**: 0.54mm (frontal), 0.51mm (profile)
- **Success Rate**: 97.96% (frontal), 99.31% (profile) at 2mm threshold
- **AUC-CED**: 1.97 (frontal), 1.99 (profile)
- **Inference Time**: ~66ms per image (CPU)

## Architecture

- **Model**: Custom CNN (4 convolutional blocks: 32→64→128→256 filters)
- **Input**: 128×128×3 RGB images
- **Output**: 22 landmarks (frontal), 15 landmarks (profile)
- **Parameters**: ~5.1M trainable
- **Framework**: TensorFlow/Keras

## Repository Contents
```
├── save_frontal_model.py      # Training script for frontal model
├── train_and_save_models.py   # Model architecture export script
├── fix_json_files.py          # JSON preprocessing utility
├── requirements.txt           # Python dependencies
└── results/                   # Experimental results
    ├── statistical_validation.csv
    ├── ced_sr_metrics.csv
    ├── auc_ced_metrics.csv
    ├── inference_timing_REAL.csv
    ├── frontal_model_comparison.csv
    └── profile_model_comparison.csv
```

## Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
python save_frontal_model.py  # Train frontal model
```

## Citation

If you use this code in your research, please cite:
```
[Your paper citation will go here after publication]
```

## Contact

For trained model weights or questions, please contact: [your email]

## License

MIT License - See LICENSE file for details

## Note

This repository contains the code and experimental results. Trained model weights are available upon reasonable request from the corresponding author due to file size limitations.

## Paper

Manuscript submitted to *Progress in Orthodontics*

**Status**: Under review
