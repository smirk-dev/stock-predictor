# 🚀 Quick Start Guide - Stock Predictor Pro

## ⚡ Fastest Way to Get Started

### Step 1: Check Your Python Version

```powershell
python --version
```

- **Python 3.13?** → Use Lite Version (Random Forest only)
- **Python 3.10-3.12?** → Use Full Version (includes LSTM/GRU)

### Step 2: Install Minimal Dependencies

**For Lite Version (Python 3.13 compatible):**
```powershell
pip install pandas numpy scikit-learn streamlit plotly matplotlib seaborn pyyaml
```

**For Full Version (Python 3.10-3.12):**
```powershell
pip install -r requirements.txt
```

### Step 3: Launch the App

**Lite Version (works with Python 3.13):**
```powershell
streamlit run app_lite.py
```

**Full Version (requires Python 3.10-3.12):**
```powershell
streamlit run app.py
```

### Step 4: Start Using!

1. 🌐 Browser opens automatically at `http://localhost:8501`
2. 📊 Go to "Data Explorer" - Select a stock (e.g., AAPL)
3. 🤖 Go to "Train Model" - Click "Start Training"
4. 🔮 Go to "Predictions" - Generate forecasts!

---

## 📋 What You Get

### Lite Version (`app_lite.py`)
- ✅ Works with **any Python version** including 3.13
- ✅ Random Forest machine learning model
- ✅ Interactive data exploration
- ✅ Technical indicators visualization
- ✅ Price predictions
- ✅ **No TensorFlow required**
- ⚡ **Faster training** (seconds vs minutes)
- 🪶 **Lighter dependencies**

### Full Version (`app.py`)
- ✅ **LSTM** deep learning (best accuracy)
- ✅ **GRU** neural network (fast deep learning)
- ✅ **Random Forest** (interpretable)
- ✅ **Ensemble** (combines all models)
- ✅ Advanced training monitoring
- ✅ More prediction options
- ⚠️ Requires Python 3.10-3.12
- ⚠️ TensorFlow/Keras required

---

## 🎯 Common Tasks

### View Stock Data
```
1. Launch: streamlit run app_lite.py
2. Navigate: Click "📊 Data Explorer"
3. Select: Choose stock ticker (AAPL, MSFT, etc.)
4. Explore: View candlestick charts, volume, statistics
```

### Train a Model
```
1. Navigate: Click "🤖 Train Model"
2. Select: Choose stock ticker
3. Configure: Set prediction horizon (1-30 days)
4. Train: Click "🚀 Start Training"
5. Wait: Progress bar shows status
6. Review: Check accuracy metrics
```

### Make Predictions
```
1. Navigate: Click "🔮 Predictions"
2. Configure: Set days ahead (1-30)
3. Predict: Click "🔮 Generate Prediction"
4. Review: See predicted price and expected change
5. Export: Download CSV results
```

---

## 🔧 Troubleshooting

### "No module named 'streamlit'"
```powershell
pip install streamlit
```

### "No module named 'src'"
```powershell
# Make sure you're in the project directory
cd C:\Users\surya\OneDrive\Desktop\suryansh\coding_projects\stock-predictor
```

### "FileNotFoundError: Data file not found"
```powershell
# Check data directory exists
dir individual_stocks_5yr\individual_stocks_5yr\AAPL_data.csv
```

### "Python 3.13 scipy/TensorFlow error"
```powershell
# Use the lite version instead
streamlit run app_lite.py
```

### "Training failed: could not convert string to float"
✅ **FIXED!** This error has been resolved. Make sure you have the latest version of `src/data/preprocessing.py`.

---

## 📊 Available Stocks

The system includes 5 years of data for 500+ S&P 500 stocks:

**Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX  
**Finance**: JPM, BAC, WFC, GS, MS, C  
**Healthcare**: JNJ, UNH, PFE, ABBV, TMO  
**Consumer**: WMT, HD, MCD, NKE, SBUX  
**Industrial**: BA, CAT, GE, MMM, HON

Full list available in `individual_stocks_5yr/individual_stocks_5yr/` directory.

---

## 🎓 Learning Path

### Beginner (10 minutes)
1. Run `streamlit run app_lite.py`
2. Explore AAPL data in Data Explorer
3. Train a quick Random Forest model
4. Make a 1-day prediction

### Intermediate (30 minutes)
1. Compare multiple stocks
2. Train models on different horizons (1, 5, 30 days)
3. Review accuracy metrics
4. Understand technical indicators

### Advanced (1+ hours)
1. Use full version with LSTM (`app.py`)
2. Train ensemble models
3. Compare model performances
4. Experiment with hyperparameters
5. Run terminal scripts (`train.py`, `predict.py`)

---

## 💻 Alternative: Terminal Commands (No GUI)

### Quick Example
```powershell
python example.py
```

### Train Specific Model
```powershell
# Random Forest (works with Python 3.13)
python train.py --ticker AAPL --model random_forest

# LSTM (requires Python 3.10-3.12)
python train.py --ticker AAPL --model lstm --epochs 50
```

### Make Predictions
```powershell
python predict.py --ticker AAPL --model models/saved_models/rf_AAPL.pkl --days 5
```

---

## 📁 Project Structure

```
stock-predictor/
├── app.py                 # Full GUI (LSTM/GRU/RF/Ensemble)
├── app_lite.py            # Lite GUI (Random Forest only) ⭐
├── train.py               # Terminal training script
├── predict.py             # Terminal prediction script
├── example.py             # Quick demo script
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── individual_stocks_5yr/ # Stock data CSV files
├── models/                # Saved models
│   ├── checkpoints/       # Training checkpoints
│   └── saved_models/      # Final models
├── src/                   # Source code
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # ML/DL models
│   ├── evaluation/        # Metrics & backtesting
│   └── utils/             # Utilities
└── outputs/               # Plots and results
```

---

## 🆘 Still Need Help?

1. **Check Installation**: `python test_app.py`
2. **Verify Setup**: `python verify_setup.py`
3. **Read Full Docs**: `README.md`
4. **View Fixes**: `FIXES.md`
5. **Installation Help**: `INSTALL.md`

---

## 🎉 You're Ready!

Pick your version and start:

```powershell
# Python 3.13 or want something simple:
streamlit run app_lite.py

# Python 3.10-3.12 and want full features:
streamlit run app.py
```

**Happy Trading! 📈**

---

*Note: This is for educational purposes only. Not financial advice.*
