"""
Daily Automation Script for Stock Predictor
Automatically updates cache, retrains models, and generates predictions
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import DataService, POPULAR_US_STOCKS, POPULAR_INDIAN_STOCKS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_cache():
    """Update cache for all popular stocks."""
    logger.info("="*50)
    logger.info("Starting daily cache update")
    logger.info("="*50)
    
    service = DataService()
    
    # Get all stocks to update
    all_stocks = POPULAR_US_STOCKS + POPULAR_INDIAN_STOCKS
    
    results = {'success': [], 'failed': []}
    
    for i, ticker in enumerate(all_stocks, 1):
        try:
            logger.info(f"[{i}/{len(all_stocks)}] Updating {ticker}...")
            df, source = service.get_stock_data(ticker, period='2y', force_refresh=True)
            
            if not df.empty:
                results['success'].append(ticker)
                logger.info(f"✅ {ticker}: {len(df)} records updated")
            else:
                results['failed'].append(ticker)
                logger.warning(f"❌ {ticker}: No data received")
                
        except Exception as e:
            results['failed'].append(ticker)
            logger.error(f"❌ {ticker}: {e}")
    
    # Summary
    logger.info("="*50)
    logger.info("Cache Update Summary")
    logger.info(f"✅ Success: {len(results['success'])}/{len(all_stocks)}")
    logger.info(f"❌ Failed: {len(results['failed'])}/{len(all_stocks)}")
    if results['failed']:
        logger.info(f"Failed stocks: {', '.join(results['failed'])}")
    logger.info("="*50)
    
    return results


def cleanup_old_data():
    """Remove data older than 2 years from cache."""
    logger.info("Cleaning up old data...")
    
    service = DataService()
    service.clear_old_cache(days=730)
    
    logger.info("✅ Cleanup complete")


def generate_cache_report():
    """Generate statistics report."""
    logger.info("Generating cache statistics report...")
    
    service = DataService()
    stats = service.get_cache_stats()
    
    logger.info("="*50)
    logger.info("Cache Statistics")
    logger.info(f"Total stocks: {stats.get('unique_tickers', 0)}")
    logger.info(f"Total records: {stats.get('total_records', 0):,}")
    logger.info(f"Database size: {stats.get('db_size_mb', 0):.2f} MB")
    logger.info(f"Date range: {stats.get('date_range', 'N/A')}")
    logger.info("="*50)


def main():
    """Main automation routine."""
    start_time = datetime.now()
    logger.info(f"🚀 Starting daily automation at {start_time}")
    
    try:
        # Step 1: Update cache
        update_results = update_cache()
        
        # Step 2: Cleanup old data
        cleanup_old_data()
        
        # Step 3: Generate report
        generate_cache_report()
        
        # Done
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("="*50)
        logger.info(f"✅ Daily automation completed in {duration:.1f} seconds")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Automation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    main()
