# Installation Guide for Stock Predictor Pro

## Python Version Compatibility

**Important:** This project is tested with Python 3.10-3.12. Python 3.13 has compatibility issues with scipy/TensorFlow.

### Recommended: Use Python 3.11 or 3.12

```powershell
# Check your Python version
python --version

# If you have Python 3.13, consider using a virtual environment with Python 3.12
# Or install Python 3.12 from python.org
```

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# Verify Python version in venv
python --version
```

### 2. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```powershell
# Install all requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```powershell
python test_app.py
```

## Common Issues

### Issue: Python 3.13 scipy/TensorFlow compatibility

**Solution:** Use Python 3.10, 3.11, or 3.12

```powershell
# Option 1: Create conda environment with specific Python version
conda create -n stock-predictor python=3.12
conda activate stock-predictor
pip install -r requirements.txt

# Option 2: Download Python 3.12 from python.org
# Then create a virtual environment
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: TensorFlow import errors

**Solution:** Ensure you're using compatible versions

```powershell
pip uninstall tensorflow keras scipy scikit-learn
pip install tensorflow==2.16.1 scipy==1.12.0 scikit-learn==1.4.2
```

### Issue: Streamlit not found

**Solution:** Reinstall streamlit

```powershell
pip install --upgrade streamlit
```

## Running the Application

Once installation is complete:

```powershell
# Run the Streamlit GUI
streamlit run app.py

# Or use python -m streamlit
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Alternative: Install Specific Compatible Versions

If you encounter issues, try these specific versions that are known to work:

```powershell
pip install pandas==2.1.4 numpy==1.24.3 scipy==1.11.4 tensorflow==2.15.0 scikit-learn==1.3.2 streamlit==1.29.0 plotly==5.18.0 matplotlib==3.8.2 seaborn==0.13.0
```

## Quick Start After Installation

1. Run verification: `python test_app.py`
2. Launch GUI: `streamlit run app.py`
3. Select a stock ticker (e.g., AAPL)
4. Explore data or train a model
5. Make predictions!

## Need Help?

- Check the README.md for full documentation
- Run `python verify_setup.py` to diagnose issues
- Ensure your data files are in the correct directories
