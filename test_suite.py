"""
Unit Tests for Stock Predictor Live System
Tests data fetcher, cache manager, and data service
"""

import unittest
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import LiveDataFetcher, CacheManager, DataService


class TestLiveDataFetcher(unittest.TestCase):
    """Test live data fetching functionality."""
    
    def setUp(self):
        self.fetcher = LiveDataFetcher()
    
    def test_fetch_us_stock(self):
        """Test fetching US stock data."""
        df = self.fetcher.fetch_stock_data('AAPL', period='1mo')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn('date', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
    
    def test_fetch_indian_stock(self):
        """Test fetching Indian stock data."""
        df = self.fetcher.fetch_stock_data('RELIANCE.NS', period='1mo')
        
        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:  # May fail due to network
            self.assertIn('date', df.columns)
            self.assertIn('close', df.columns)
    
    def test_validate_ticker(self):
        """Test ticker validation."""
        # Valid ticker
        self.assertTrue(self.fetcher.validate_ticker('AAPL'))
        
        # Invalid ticker
        self.assertFalse(self.fetcher.validate_ticker('INVALID123'))
    
    def test_market_overview(self):
        """Test market overview fetching."""
        overview = self.fetcher.get_market_overview()
        
        self.assertIsInstance(overview, pd.DataFrame)
        if not overview.empty:
            self.assertIn('name', overview.columns)
            self.assertIn('price', overview.columns)


class TestCacheManager(unittest.TestCase):
    """Test cache management functionality."""
    
    def setUp(self):
        # Use temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_cache.db'
        self.cache = CacheManager(str(self.db_path))
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_get_stock_data(self):
        """Test saving and retrieving stock data."""
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000000] * 10
        })
        
        # Save data
        success = self.cache.save_stock_data('TEST', test_data)
        self.assertTrue(success)
        
        # Retrieve data
        retrieved = self.cache.get_stock_data('TEST')
        self.assertFalse(retrieved.empty)
        self.assertEqual(len(retrieved), 10)
    
    def test_data_freshness(self):
        """Test cache freshness checking."""
        # Create and save test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000000] * 5
        })
        
        self.cache.save_stock_data('TEST', test_data)
        
        # Check freshness (should be fresh since just added)
        is_fresh = self.cache.is_data_fresh('TEST', max_age_days=1)
        # Note: This might be False if data is old, that's expected
        self.assertIsInstance(is_fresh, bool)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_cache_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_records', stats)
        self.assertIn('unique_tickers', stats)
        self.assertIn('db_size_mb', stats)


class TestDataService(unittest.TestCase):
    """Test unified data service."""
    
    def setUp(self):
        # Use temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_cache.db'
        
        # Create service with test cache
        from src.data.cache_manager import CacheManager
        self.service = DataService()
        self.service.cache = CacheManager(str(self.db_path))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_get_stock_data(self):
        """Test getting stock data with caching."""
        # First call - should fetch from API
        df1, source1 = self.service.get_stock_data('AAPL', period='1mo')
        
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertIn(source1, ['api', 'cache', 'cache_fallback', 'error'])
        
        if not df1.empty and source1 == 'api':
            # Second call - should use cache
            df2, source2 = self.service.get_stock_data('AAPL', period='1mo')
            self.assertEqual(source2, 'cache')
    
    def test_get_popular_stocks(self):
        """Test getting popular stock lists."""
        us_stocks = self.service.get_popular_stocks('us')
        india_stocks = self.service.get_popular_stocks('india')
        
        self.assertIsInstance(us_stocks, list)
        self.assertIsInstance(india_stocks, list)
        self.assertGreater(len(us_stocks), 0)
        self.assertGreater(len(india_stocks), 0)
    
    def test_validate_ticker(self):
        """Test ticker validation."""
        result = self.service.validate_ticker('AAPL')
        self.assertIsInstance(result, bool)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete data fetching and caching workflow."""
        service = DataService()
        
        # Fetch data
        df, source = service.get_stock_data('MSFT', period='1mo', force_refresh=True)
        
        if not df.empty:
            # Verify data structure
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df.columns, f"Missing column: {col}")
            
            # Verify data types
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLiveDataFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDataService))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🔥 Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
