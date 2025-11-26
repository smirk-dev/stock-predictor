# Models Directory

This directory contains trained models and training checkpoints.

## Structure

```
models/
├── checkpoints/       # Training checkpoints (saved during training)
│   ├── lstm_best.keras
│   ├── gru_best.keras
│   └── ...
│
└── saved_models/      # Final trained models
    ├── AAPL_lstm_model.keras
    ├── AAPL_gru_model.keras
    ├── AAPL_random_forest_model.pkl
    └── ...
```

## Model Files

### Deep Learning Models (.keras)
- LSTM and GRU models saved in Keras format
- Can be loaded with: `model.load_model('path/to/model.keras')`
- Includes architecture and trained weights

### Baseline Models (.pkl)
- Random Forest and other scikit-learn models
- Saved using joblib
- Can be loaded with: `joblib.load('path/to/model.pkl')`

## Usage

**Load a model:**
```python
from src.models.lstm_model import LSTMPredictor

model = LSTMPredictor()
model.load_model('models/saved_models/AAPL_lstm_model.keras')
```

**Make predictions:**
```python
predictions = model.predict(X_test)
```

## Checkpoints

During training, the best model (based on validation loss) is automatically saved to `checkpoints/`.

These checkpoints are used to restore the best weights after training completes.
