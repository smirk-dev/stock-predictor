# 🚀 Quick Setup Guide

## Installation Steps

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- TensorFlow/Keras for deep learning
- scikit-learn for preprocessing
- pandas/numpy for data manipulation
- matplotlib/seaborn/plotly for visualization
- Technical analysis libraries (ta, pandas-ta)

### 2. Verify Installation

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

### 3. Test the Setup

```powershell
# Quick test - train a simple model on AAPL (first 100 epochs)
python train.py --ticker AAPL --model lstm
```

## 📁 Data Structure

Ensure your data is organized as:
```
stock-predictor/
├── all_stocks_5yr.csv
└── individual_stocks_5yr/
    └── individual_stocks_5yr/
        ├── AAPL_data.csv
        ├── MSFT_data.csv
        ├── GOOGL_data.csv
        └── ...
```

## 🎯 First Steps

### Step 1: Explore the Data

```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- Load and validate your stock data
- Show statistical summaries
- Visualize price trends
- Analyze volatility

### Step 2: Train Your First Model

```powershell
# Train LSTM on Apple stock
python main.py train --ticker AAPL --model lstm
```

This will:
- Load AAPL data
- Calculate 20+ technical indicators
- Train an LSTM model
- Save the model to `models/saved_models/`
- Generate performance plots in `outputs/plots/`

### Step 3: Make Predictions

```powershell
# Predict next day's price
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras
```

This will:
- Load the trained model
- Make predictions for tomorrow
- Show predicted price and direction
- Save results to `outputs/`

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Example: Change LSTM architecture
models:
  lstm:
    units: [256, 128, 64]  # More layers, more complexity
    dropout: 0.3           # Higher dropout for regularization
    learning_rate: 0.0005  # Lower learning rate
    epochs: 150            # Train longer
```

## 📊 Understanding the Outputs

After training, you'll find:

**`models/saved_models/`** - Trained models
- `AAPL_lstm_model.keras` - Saved LSTM model

**`outputs/plots/`** - Visualizations
- `AAPL_lstm_predictions.png` - Predictions vs actual
- `AAPL_lstm_history.png` - Training loss curves
- `AAPL_lstm_errors.png` - Error distribution

**`logs/`** - Training logs
- `training.log` - Detailed training information

## 🎓 Learning Path

**Beginner:**
1. Run data exploration notebook
2. Train LSTM on single stock
3. Make predictions
4. Understand the metrics

**Intermediate:**
5. Experiment with different technical indicators
6. Try GRU and Random Forest models
7. Run backtesting strategies
8. Optimize hyperparameters

**Advanced:**
9. Train ensemble model
10. Implement custom features
11. Create custom trading strategies
12. Deploy for real-time predictions

## 🐛 Common Issues

### Issue: ModuleNotFoundError

```powershell
# Solution: Install missing package
pip install <package-name>
```

### Issue: CUDA/GPU not detected

```powershell
# TensorFlow will automatically use CPU
# For GPU support, install:
pip install tensorflow-gpu
```

### Issue: Memory error during training

```yaml
# Solution: Reduce batch size in config.yaml
models:
  lstm:
    batch_size: 16  # Reduced from 32
```

## 📚 Next Steps

1. **Experiment**: Try different stocks and models
2. **Customize**: Add your own technical indicators
3. **Evaluate**: Compare model performance
4. **Deploy**: Use best model for predictions

## 💡 Tips

- Start with single stock before training on all stocks
- Use shorter sequences (30 days) for faster training
- LSTM is most accurate, GRU is fastest
- Ensemble model gives best results but takes longest

## 🔗 Useful Commands

```powershell
# Train different models
python main.py train --ticker AAPL --model lstm
python main.py train --ticker AAPL --model gru
python main.py train --ticker AAPL --model random_forest

# Train on all stocks
python main.py train --model ensemble

# Predict multiple horizons
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras --days 1
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras --days 5
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras --days 30

# Launch Jupyter
jupyter notebook
```

## ✅ Verification Checklist

- [ ] All dependencies installed
- [ ] Data files in correct location
- [ ] Successfully loaded a stock dataset
- [ ] Trained a model (even on small data)
- [ ] Generated a prediction
- [ ] Opened a Jupyter notebook

## 🎉 You're Ready!

You now have a complete stock prediction system. Start exploring and happy predicting!

**Remember**: This is for educational purposes. Always do your own research before making investment decisions.
