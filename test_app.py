"""
Quick test script to verify setup before running Streamlit app.
"""

import sys
print("Python version:", sys.version)

# Test imports
try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print("✗ pandas import failed:", e)

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print("✗ numpy import failed:", e)

try:
    import tensorflow as tf
    print(f"✓ tensorflow {tf.__version__} imported successfully")
except ImportError as e:
    print("✗ tensorflow import failed:", e)

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__} imported successfully")
except ImportError as e:
    print("✗ scikit-learn import failed:", e)

try:
    import streamlit as st
    print(f"✓ streamlit {st.__version__} imported successfully")
except ImportError as e:
    print("✗ streamlit import failed:", e)

try:
    import plotly
    print(f"✓ plotly {plotly.__version__} imported successfully")
except ImportError as e:
    print("✗ plotly import failed:", e)

try:
    from src.config import get_config
    config = get_config()
    print("✓ src.config imported successfully")
    print(f"  Project root: {config.project_root}")
except Exception as e:
    print("✗ src.config import failed:", e)

try:
    from src.data.data_loader import StockDataLoader
    print("✓ StockDataLoader imported successfully")
except Exception as e:
    print("✗ StockDataLoader import failed:", e)

print("\nSetup check complete!")
