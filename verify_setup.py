"""
Quick start script to verify installation and run a simple example.
"""

import sys
from pathlib import Path

def check_installation():
    """Check if all required packages are installed."""
    print("="*80)
    print("CHECKING INSTALLATION")
    print("="*80)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml',
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"✗ {pip_name} is NOT installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("\n" + "="*80)
        print("MISSING PACKAGES")
        print("="*80)
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All required packages are installed!")
    return True


def check_data_files():
    """Check if data files exist."""
    print("\n" + "="*80)
    print("CHECKING DATA FILES")
    print("="*80)
    
    data_files = [
        'all_stocks_5yr.csv',
        'individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv'
    ]
    
    all_found = True
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_found = False
    
    if not all_found:
        print("\n⚠ Some data files are missing.")
        print("Please ensure your data files are in the correct location.")
    else:
        print("\n✓ All data files found!")
    
    return all_found


def run_quick_test():
    """Run a quick test of the system."""
    print("\n" + "="*80)
    print("RUNNING QUICK TEST")
    print("="*80)
    
    try:
        from src.config import get_config
        from src.data.data_loader import StockDataLoader
        
        print("\n1. Loading configuration...")
        config = get_config()
        print("   ✓ Configuration loaded successfully")
        
        print("\n2. Loading data...")
        loader = StockDataLoader(config.project_root)
        
        try:
            df = loader.load_individual_stock('AAPL', 
                config.get_path('data.individual_stocks_dir'))
            print(f"   ✓ Loaded AAPL data: {len(df)} records")
            print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        except FileNotFoundError:
            print("   ⚠ AAPL data file not found")
            return False
        
        print("\n3. Testing feature engineering...")
        from src.data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        df_features = engineer.add_all_features(df.head(100), config.feature_config)
        print(f"   ✓ Added features. Total columns: {len(df_features.columns)}")
        
        print("\n✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "STOCK PREDICTOR - SETUP VERIFICATION" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Check installation
    if not check_installation():
        print("\n" + "="*80)
        print("Please install missing packages and run this script again.")
        print("="*80)
        sys.exit(1)
    
    # Check data files
    data_ok = check_data_files()
    
    # Run quick test
    if data_ok:
        test_ok = run_quick_test()
    else:
        test_ok = False
    
    # Final summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    print(f"Packages: {'✓ OK' if True else '✗ Issues'}")
    print(f"Data Files: {'✓ OK' if data_ok else '⚠ Some missing'}")
    print(f"System Test: {'✓ PASSED' if test_ok else '✗ FAILED'}")
    print("="*80)
    
    if test_ok:
        print("\n🎉 Your setup is complete! You're ready to start.")
        print("\nNext steps:")
        print("  1. Train a model:")
        print("     python main.py train --ticker AAPL --model lstm")
        print("\n  2. Or explore with Jupyter:")
        print("     jupyter notebook notebooks/01_data_exploration.ipynb")
        print("\n  3. Read the documentation:")
        print("     See README.md and SETUP.md for detailed instructions")
    else:
        print("\n⚠ Setup incomplete. Please resolve the issues above.")
        print("See SETUP.md for troubleshooting tips.")
    
    print("\n")


if __name__ == "__main__":
    main()
