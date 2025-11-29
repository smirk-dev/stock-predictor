"""Quick test of new data modules"""

from src.data import DataService
import pandas as pd

print("Testing DataService...")

# Initialize service
service = DataService()
print("✅ DataService initialized")

# Test market overview
try:
    overview = service.get_market_overview()
    print(f"✅ Market overview: {len(overview)} indices loaded")
    if not overview.empty:
        print(overview[['name', 'price', 'change_pct']].head())
except Exception as e:
    print(f"❌ Market overview error: {e}")

# Test stock loading
try:
    print("\nTesting stock data loading...")
    df, source = service.get_stock_data('AAPL', period='1y')
    print(f"✅ Loaded AAPL: {len(df)} records from {source}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"❌ Stock loading error: {e}")

# Test cache stats
try:
    stats = service.get_cache_stats()
    print(f"\n✅ Cache stats:")
    print(f"   Cached stocks: {stats.get('unique_tickers', 0)}")
    print(f"   Total records: {stats.get('total_records', 0):,}")
    print(f"   DB size: {stats.get('db_size_mb', 0):.2f} MB")
except Exception as e:
    print(f"❌ Cache stats error: {e}")

print("\n🎉 All tests completed!")
