# 🚀 Quick Start - Live Market Edition

## Prerequisites
- Python virtual environment activated
- All dependencies installed

## 1. Activate Environment
```powershell
.\stock\Scripts\Activate.ps1
```

## 2. Launch App
```powershell
streamlit run app_live.py --server.port 8502
```

## 3. Access App
Open browser: **http://localhost:8502**

## 4. Load Your First Stock

### Option A: Popular US Stocks
1. Go to "🔍 Stock Search" tab
2. Select from **Popular US Stocks** dropdown (e.g., AAPL)
3. Click **"📥 Load Stock Data"**

### Option B: Indian Stocks
1. Select from **Popular Indian Stocks** (e.g., RELIANCE.NS)
2. Click **"📥 Load Stock Data"**

### Option C: Any Stock
1. Enter ticker in **"Or enter ticker directly"** field
   - US stocks: `MSFT`, `GOOGL`, `TSLA`
   - Indian stocks: `TCS.NS`, `INFY.NS` (NSE) or `TCS.BO` (BSE)
2. Choose period: 1y, 2y, 5y, or max
3. Click **"📥 Load Stock Data"**

## 5. View Analysis
1. Click **"📊 Analysis"** tab
2. See candlestick chart, volume, and statistics

## 6. Check Cache
- Sidebar shows cache statistics
- First load: fetches from API (🔴 LIVE)
- Subsequent loads: uses cache (💾 CACHED)

## That's It! 🎉

Your stock predictor now has:
- ✅ Real-time data
- ✅ Global market access  
- ✅ Automatic caching
- ✅ Unlimited stocks

## Next Steps
- Load multiple stocks to populate cache
- Train models with fresh data (coming in Phase 4)
- Make predictions (coming in Phase 4)

## Troubleshooting

**App won't start?**
```powershell
.\stock\Scripts\pip.exe install streamlit plotly yfinance pyyaml
```

**Can't load stocks?**
- Check internet connection
- Try a different ticker
- Check if ticker exists on Yahoo Finance

**Need help?**
Read `TRANSFORMATION_GUIDE.md` for full documentation.
