# Stock Predictor Pro - Error Fixes and Solutions

## Latest Fix (Matrix Size Mismatch)

### ✅ Fixed LSTM Prediction Error
**Error**: `Matrix size-incompatible: In[0]: [1,39], In[1]: [40,512]`

**Problem**: When making predictions, the model was receiving a different number of features than it was trained with.

**Solution**: 
1. Store the exact feature list and scaler used during training
2. Use the same scaler.transform() for prediction (not scale_features() which creates a new scaler)
3. Properly reshape input for LSTM: `(1, sequence_length, n_features)`
4. Verify feature count matches before prediction

**Changes in app.py**:
```python
# When saving trained model:
'scaler': preprocessor.feature_scaler,
'n_features': len(feature_cols)

# During prediction:
recent_scaled[feature_cols] = trained_info['preprocessor'].feature_scaler.transform(
    recent_data[feature_cols]
)
X_pred = last_sequence.reshape(1, sequence_length, len(feature_cols))
```

---

## Summary of All Fixes

### 1. ✅ Fixed "could not convert string to float: 'AAPL'" Error

**Problem**: The 'name'/'Name' column was being included in numerical feature scaling.

**Solution**: Updated three functions in `src/data/preprocessing.py`:
- `scale_features()`: Excludes both 'name' and 'Name' columns, only includes numeric dtype columns
- `create_sequences()`: Excludes both 'name' and 'Name' columns, checks for numeric dtypes
- `get_feature_importance_columns()`: Excludes both variants and filters for numeric columns only

**Changes**:
```python
# Before
exclude_cols = ['date', 'name', 'target', 'target_direction']

# After
exclude_cols = ['date', 'name', 'Name', 'target', 'target_direction']
feature_cols = [col for col in df.columns 
                if col not in exclude_cols 
                and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
```

### 2. ✅ Fixed Variable Scoping Issues in Streamlit App

**Problem**: Variables `learning_rate`, `batch_size`, `epochs`, `model`, and `history` were possibly unbound.

**Solution**: 
- Initialized default values for hyperparameters outside conditional blocks
- Initialized `model` and `history` to `None` before training loop
- Added null check before model evaluation

### 3. ✅ Standardized Column Names

**Problem**: Inconsistent use of 'Name' vs 'name' in data loading.

**Solution**: Updated `src/data/data_loader.py` to:
- Convert 'Name' to lowercase 'name' when loading
- Consistently use 'name' throughout the codebase

### 4. ✅ Python 3.13 Compatibility Issue

**Problem**: Python 3.13 has compatibility issues with scipy/TensorFlow.

**Solution**:
- Updated `requirements.txt` with version constraints
- Created `INSTALL.md` with workaround instructions
- Created `app_lite.py` - lightweight version using only Random Forest (no TensorFlow)

## Files Modified

1. **app.py** - Fixed variable scoping, added model null checks
2. **src/data/preprocessing.py** - Fixed column filtering in 3 functions
3. **src/data/data_loader.py** - Standardized column names
4. **requirements.txt** - Added version constraints for Python 3.13 compatibility
5. **INSTALL.md** (new) - Comprehensive installation guide with troubleshooting
6. **app_lite.py** (new) - Lightweight version without deep learning
7. **test_app.py** (new) - Quick verification script

## How to Use

### Option 1: Use Lite Version (Recommended for Python 3.13)

```powershell
streamlit run app_lite.py
```

Features:
- ✅ Works with Python 3.13
- ✅ No TensorFlow/scipy issues
- ✅ Random Forest predictions
- ✅ All data exploration features
- ❌ No LSTM/GRU models

### Option 2: Use Full Version (Python 3.10-3.12)

```powershell
# Install Python 3.12 or use conda
conda create -n stock-predictor python=3.12
conda activate stock-predictor
pip install -r requirements.txt
streamlit run app.py
```

Features:
- ✅ LSTM deep learning
- ✅ GRU models
- ✅ Random Forest
- ✅ Ensemble predictions
- ✅ All features

### Option 3: Use Terminal Scripts (No GUI)

```powershell
# Train a model
python train.py --ticker AAPL --model lstm

# Make predictions
python predict.py --ticker AAPL --model models/saved_models/lstm_AAPL.keras --days 5

# Quick example
python example.py
```

## Testing Your Setup

```powershell
# 1. Test imports and data loading
python test_app.py

# 2. Verify complete setup
python verify_setup.py

# 3. Run lite GUI (no deep learning)
streamlit run app_lite.py

# 4. Run full GUI (if Python 3.10-3.12)
streamlit run app.py
```

## Error Resolutions

### ✅ "could not convert string to float"
**Status**: FIXED
**Solution**: Column filtering now excludes string columns and checks dtypes

### ✅ "model is possibly unbound"
**Status**: FIXED  
**Solution**: Variables initialized before conditional blocks

### ✅ "Python 3.13 scipy import error"
**Status**: WORKAROUND PROVIDED
**Solution**: Use `app_lite.py` or downgrade to Python 3.12

### ✅ "No prediction model"
**Status**: FIXED
**Solution**: Both `app.py` and `app_lite.py` have full train & predict functionality

## Quick Start Commands

```powershell
# Install dependencies (lite version - works with Python 3.13)
pip install pandas numpy scikit-learn streamlit plotly matplotlib seaborn pyyaml

# Run lite version
streamlit run app_lite.py

# For full version with deep learning (Python 3.12):
# 1. Use Python 3.12
# 2. pip install -r requirements.txt
# 3. streamlit run app.py
```

## All Fixed Issues Summary

| Issue | Status | File(s) Modified |
|-------|--------|-----------------|
| String to float conversion error | ✅ Fixed | preprocessing.py |
| Variable scoping in training | ✅ Fixed | app.py |
| Column name inconsistency | ✅ Fixed | data_loader.py |
| Python 3.13 compatibility | ✅ Workaround | app_lite.py, INSTALL.md |
| Missing prediction functionality | ✅ Added | app.py, app_lite.py |
| Model evaluation errors | ✅ Fixed | app.py |

## Next Steps

1. **Test Lite Version**: `streamlit run app_lite.py`
2. **Verify Data**: Check that `individual_stocks_5yr/individual_stocks_5yr/` has CSV files
3. **Train Model**: Select a stock and click "Start Training"
4. **Make Predictions**: After training, go to Predictions page
5. **Explore Data**: View charts and technical indicators

All major errors have been resolved! 🎉
