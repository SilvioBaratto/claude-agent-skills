---
name: yfinance
description: |
  Load proactively whenever the user works with yfinance or Yahoo Finance data. Do not wait to be asked; apply this skill automatically. Covers Ticker/Tickers objects, price history (history/download), financial statements, analyst recommendations, screeners, websockets, sector/industry data, search/lookup, calendars, fund data, market status, configuration, caching, price repair, and multi-level column handling.
---

# yfinance API Skill

Expert guidance for using yfinance — a Python library for retrieving financial data from Yahoo Finance. Covers all public APIs: Ticker, Tickers, download, Market, Search, Lookup, Screener, WebSocket, Sector, Industry, Calendars, FundsData, and configuration.

## Documentation Reference

When in doubt, **always consult** the official documentation:

| Topic | URL |
|-------|-----|
| **Advanced** | |
| Price History | https://ranaroussi.github.io/yfinance/advanced/price-history.html |
| Financials Data | https://ranaroussi.github.io/yfinance/advanced/financials.html |
| Stock Info | https://ranaroussi.github.io/yfinance/advanced/stock.html |
| Analysis Data | https://ranaroussi.github.io/yfinance/advanced/analysis.html |
| Funds Data | https://ranaroussi.github.io/yfinance/advanced/funds.html |
| Sector & Industry | https://ranaroussi.github.io/yfinance/advanced/sector-industry.html |
| Market Data | https://ranaroussi.github.io/yfinance/advanced/market.html |
| Search & Lookup | https://ranaroussi.github.io/yfinance/advanced/search.html |
| Calendars | https://ranaroussi.github.io/yfinance/advanced/calendars.html |
| Screener | https://ranaroussi.github.io/yfinance/advanced/screener.html |
| WebSocket | https://ranaroussi.github.io/yfinance/advanced/websocket.html |
| **Reference** | |
| Ticker API | https://ranaroussi.github.io/yfinance/reference/ticker/index.html |
| Tickers API | https://ranaroussi.github.io/yfinance/reference/tickers.html |
| Download Function | https://ranaroussi.github.io/yfinance/reference/functions/download.html |
| Market API | https://ranaroussi.github.io/yfinance/reference/market.html |
| Search API | https://ranaroussi.github.io/yfinance/reference/search.html |
| Screener API | https://ranaroussi.github.io/yfinance/reference/screener.html |
| Sector & Industry API | https://ranaroussi.github.io/yfinance/reference/sector-industry.html |

## Architecture Overview

```
yfinance/
├── ticker.py           # Ticker class — central entry point
├── stock.py            # Stock info, fast_info, news, ISIN
├── market.py           # Market status and summary
├── financials.py       # Income stmt, balance sheet, cash flow
├── analysis.py         # Recommendations, price targets, estimates
├── price_history.py    # history(), download()
├── search.py           # Search and Lookup classes
├── screener.py         # EquityQuery, FundQuery, screen()
├── websocket.py        # WebSocket, AsyncWebSocket
├── sector_industry.py  # Sector, Industry classes
├── calendars.py        # Calendars (earnings, IPOs, splits)
├── funds_data.py       # Fund-specific data (holdings, weightings)
└── functions.py        # Module-level download() helper
```

## Complete API Reference

### Ticker and Tickers

```python
import yfinance as yf

# Single ticker
ticker = yf.Ticker("AAPL", session=None)

# Multiple tickers
tickers = yf.Tickers("AAPL MSFT GOOG", session=None)
# Or: yf.Tickers(["AAPL", "MSFT", "GOOG"])
# Access individual: tickers.tickers["AAPL"].info
```

### yf.download()

Bulk price data retrieval. Returns a DataFrame (single ticker) or multi-level DataFrame (multiple tickers).

```python
data = yf.download(
    tickers,              # str or list — "AAPL" or ["AAPL", "MSFT"]
    period="1mo",         # Valid: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval="1d",        # Valid: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    start=None,           # str or datetime — "2020-01-01"
    end=None,             # str or datetime — "2024-01-01"
    group_by="column",    # "column" (default) or "ticker"
    auto_adjust=True,     # Adjust OHLC for splits/dividends
    repair=False,         # Repair known data issues
    actions=True,         # Include dividends and stock splits
    threads=True,         # Multi-threaded download
    proxy=None,           # Proxy URL
    progress=True,        # Show progress bar
    timeout=10,           # Request timeout in seconds
    multi_level_index=True,  # Multi-level columns for multi-ticker
)
```

**Column order:** `Open, High, Low, Close, Volume` (auto_adjust=True removes Adj Close).

### yf.Market

```python
market = yf.Market("us_market", session=None, timeout=10)
market.status     # dict — market open/close status
market.summary    # dict — market summary data
```

**Valid market identifiers:** `us_market`, `gb_market`, `de_market`, `fr_market`, `jp_market`, `hk_market`, `ca_market`, `in_market`, `br_market`, `au_market`

### yf.Search and yf.Lookup

```python
# Search — full-text across quotes and news
search = yf.Search(
    query="Apple",
    max_results=8,         # default 8
    news_count=8,          # default 8
    enable_fuzzy_query=False,
    session=None,
    timeout=10,
)
search.quotes    # list[dict] — matching symbols
search.news      # list[dict] — related news articles
search.research  # list[dict] — research reports

# Lookup — screen-like symbol lookup
lookup = yf.Lookup(
    query="tech",
    type="equity",         # "equity", "mutualfund", "etf", "index", "future", "currency"
    session=None,
    timeout=10,
)
lookup.quotes    # list[dict] — matching symbols
```

### yf.Calendars

```python
cal = yf.Calendars(
    start="2024-01-01",   # str or date
    end="2024-12-31",
    session=None,
)
cal.earnings     # DataFrame — earnings calendar
cal.ipos         # DataFrame — IPO calendar
cal.splits       # DataFrame — stock splits
cal.economic_events  # DataFrame — economic events
```

### yf.Sector and yf.Industry

```python
sector = yf.Sector(key="technology", session=None)
sector.overview       # dict — sector overview
sector.top_companies  # DataFrame — top companies in sector
sector.industries     # DataFrame — industries in sector
sector.top_etfs       # DataFrame — sector-tracking ETFs
sector.research       # list — research reports

industry = yf.Industry(key="semiconductors", session=None)
industry.overview       # dict — industry overview
industry.top_companies  # DataFrame — top companies
industry.top_etfs       # DataFrame — industry-tracking ETFs
industry.research       # list — research reports
```

**Valid sector keys:** `technology`, `healthcare`, `financial-services`, `consumer-cyclical`, `communication-services`, `industrials`, `consumer-defensive`, `energy`, `basic-materials`, `real-estate`, `utilities`

### Screener API

```python
from yfinance import EquityQuery, FundQuery, screen

# Build query with operators
query = EquityQuery("and", [
    EquityQuery("gt", ["marketcap", 1_000_000_000]),      # > $1B market cap
    EquityQuery("lt", ["peratio", 20]),                     # P/E < 20
    EquityQuery("eq", ["sector", "Technology"]),            # Tech sector
])

# Or nested AND/OR
query = EquityQuery("or", [
    EquityQuery("and", [
        EquityQuery("gt", ["intradayprice", 50]),
        EquityQuery("lt", ["intradayprice", 200]),
    ]),
    EquityQuery("gt", ["dividendyield", 3]),
])

# Execute screen
result = yf.screen(
    query,
    sort_field="marketcap",
    sort_type="desc",
    offset=0,
    size=25,              # max 250
)
# result["quotes"] — list of matching stocks
# result["total"] — total matches

# Fund screening
fund_query = FundQuery("and", [
    FundQuery("gt", ["netassets", 1_000_000_000]),
    FundQuery("lt", ["annual_return_nav_y5", 10]),
])
```

**Valid EquityQuery fields (categories):**
- **Valuation:** `marketcap`, `peratio`, `pbratio`, `enterprisevalue`, `evtoebitda`, `evtorevenue`, `pricetosales`
- **Price:** `intradayprice`, `intradaymarketcap`, `fiftytwowkhigh`, `fiftytwoweeklow`, `intradaypricepctchange`
- **Dividends:** `dividendyield`, `trailingannualdividendyield`, `payoutratio`
- **Financials:** `revenue`, `ebitda`, `netincome`, `totaldebt`, `totalcash`, `grossprofitmargin`, `operatingmargin`, `profitmargin`
- **Growth:** `revenuegrowthquarterly`, `earningsgrowthquarterly`
- **Classification:** `sector`, `industry`, `exchange`, `region`
- **Analyst:** `recommendationkey`, `numberofanalystopinions`, `targetmeanprice`

**Valid operators:** `gt` (>), `lt` (<), `eq` (=), `gte` (>=), `lte` (<=), `btwn` (between), `and`, `or`

### WebSocket

```python
# Synchronous
def on_message(ws, msg):
    print(msg)

ws = yf.WebSocket()
ws.subscribe(["AAPL", "MSFT"])
ws.on_message = on_message
ws.run()  # blocks

# Asynchronous
import asyncio

async def main():
    ws = yf.AsyncWebSocket()
    ws.subscribe(["AAPL", "MSFT"])

    async for msg in ws:
        print(msg)

asyncio.run(main())
```

### Stock Properties

Accessed via `yf.Ticker(symbol)`:

| Property | Returns | Description |
|----------|---------|-------------|
| `info` | dict | Complete stock info (slow, cached) |
| `fast_info` | FastInfo | Key metrics (fast, fewer fields) |
| `news` | list[dict] | Recent news articles |
| `dividends` | Series | Historical dividends |
| `splits` | Series | Historical stock splits |
| `actions` | DataFrame | Dividends + splits combined |
| `capital_gains` | Series | Capital gains distributions (funds) |
| `shares_full` | DataFrame | Historical shares outstanding |
| `get_shares_full()` | DataFrame | Same as above, with start/end params |
| `isin` | str | ISIN identifier |
| `options` | tuple | Available option expiry dates |
| `option_chain(date)` | OptionChain | Calls and puts for expiry date |

**fast_info fields:** `currency`, `dayHigh`, `dayLow`, `exchange`, `fiftyDayAverage`, `lastPrice`, `lastVolume`, `marketCap`, `open`, `previousClose`, `quoteType`, `regularMarketPreviousClose`, `shares`, `tenDayAverageVolume`, `threeMonthAverageVolume`, `timezone`, `twoHundredDayAverage`, `yearChange`, `yearHigh`, `yearLow`

### Financial Statements

All accept `freq` param: `"yearly"` (default), `"quarterly"`, or `"trailing"` (TTM).

```python
ticker = yf.Ticker("AAPL")

# Income statement
ticker.income_stmt              # Annual
ticker.quarterly_income_stmt    # Quarterly
ticker.get_income_stmt(freq="trailing", as_dict=False, pretty=False)

# Balance sheet
ticker.balance_sheet
ticker.quarterly_balance_sheet
ticker.get_balance_sheet(freq="quarterly")

# Cash flow
ticker.cashflow
ticker.quarterly_cashflow
ticker.get_cashflow(freq="trailing")

# Earnings (simplified income)
ticker.earnings
ticker.quarterly_earnings

# SEC filings
ticker.sec_filings    # list[dict] — recent SEC filings
```

### Analysis Data

```python
ticker = yf.Ticker("AAPL")

# Analyst recommendations
ticker.recommendations           # DataFrame — recent recommendations
ticker.recommendations_summary   # DataFrame — summary (Buy/Hold/Sell counts)
ticker.upgrades_downgrades       # DataFrame — upgrades and downgrades

# Price targets
ticker.analyst_price_targets     # dict — current/low/high/mean/median

# Earnings estimates
ticker.earnings_estimate         # DataFrame — current/next quarter estimates
ticker.revenue_estimate          # DataFrame — revenue estimates
ticker.earnings_history          # DataFrame — EPS surprise history
ticker.eps_trend                 # DataFrame — EPS trend (current vs 7/30/60/90 days ago)
ticker.eps_revisions             # DataFrame — EPS revision counts (up/down)
ticker.growth_estimates          # DataFrame — growth estimates vs sector/industry

# ESG / Sustainability
ticker.sustainability            # DataFrame — ESG scores

# Ownership
ticker.major_holders             # DataFrame — % held by insiders/institutions
ticker.institutional_holders     # DataFrame — top institutional holders
ticker.mutualfund_holders        # DataFrame — top mutual fund holders
ticker.insider_transactions      # DataFrame — recent insider transactions
ticker.insider_purchases         # DataFrame — insider purchase summary
ticker.insider_roster_holders    # DataFrame — insider roster
```

### Price History

```python
ticker = yf.Ticker("AAPL")

history = ticker.history(
    period="1mo",        # Valid: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval="1d",       # Valid: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    start=None,          # str or datetime
    end=None,            # str or datetime
    prepost=False,       # Include pre/post market data
    auto_adjust=True,    # Adjust for splits/dividends
    repair=False,        # Repair known data issues
    keepna=False,        # Keep NaN rows
    rounding=False,      # Round prices to 2 decimals
    raise_errors=False,  # Raise exceptions vs return empty
)
# Returns DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
```

**Period-interval constraints:**
- Intraday intervals (1m-90m): max 60 days of data
- 1m interval: max 7 days
- 1h interval: max 730 days
- 1d and above: full history available

### FundsData Properties

For ETFs and mutual funds:

```python
ticker = yf.Ticker("SPY")

ticker.fund_overview             # dict — fund description, family, category
ticker.fund_top_holdings         # DataFrame — top holdings with % weight
ticker.fund_sector_weightings    # DataFrame — sector allocation
ticker.fund_asset_allocation     # dict — stocks/bonds/cash/other %
ticker.fund_performance          # dict — trailing/annual returns
ticker.fund_holding_info         # dict — turnover, inception date, AUM
ticker.fund_equity_holdings      # dict — P/E, P/B, price/sales of holdings
ticker.fund_bond_holdings        # dict — duration, credit quality, maturity
ticker.fund_bond_ratings         # dict — credit rating breakdown
```

---

## Advanced Configuration

### Network Configuration

```python
import yfinance as yf

# Proxy
yf.config.network.proxy = "http://proxy:8080"

# Retries
yf.config.network.retries = 5
```

### Debug Configuration

```python
# Hide exceptions (return empty results instead)
yf.config.debug.hide_exceptions = True

# Enable logging
yf.config.debug.logging = True

# Full debug mode
yf.enable_debug_mode()
```

### Caching

yfinance caches timezone data by default.

```python
# Set custom cache location
yf.set_tz_cache_location("/path/to/cache")
```

**Default cache paths:**
- macOS: `~/Library/Caches/py-yfinance`
- Linux: `~/.cache/py-yfinance`
- Windows: `%LOCALAPPDATA%\py-yfinance\Cache`

### Price Repair

When `repair=True`, yfinance detects and fixes:

1. **Missing dividend adjustment** — prices not adjusted after dividend
2. **Missing stock split adjustment** — prices not adjusted after split
3. **Missing data** — gaps filled from adjacent intervals
4. **Corrupt data** — outlier detection and replacement
5. **100x currency errors** — wrong currency unit (e.g., pence vs pounds)
6. **Dividend repair** — incorrect dividend amounts

```python
df = ticker.history(period="2y", repair=True)
# Check what was repaired:
if "Repaired?" in df.columns:
    repaired = df[df["Repaired?"] == True]
    print(f"Repaired {len(repaired)} rows")
```

### Multi-Level Columns

When downloading multiple tickers, yfinance returns multi-level columns.

```python
# group_by='column' (default): Level 0 = Price/Volume, Level 1 = Ticker
data = yf.download(["AAPL", "MSFT"], group_by="column")
# Access: data["Close"]["AAPL"]

# group_by='ticker': Level 0 = Ticker, Level 1 = Price/Volume
data = yf.download(["AAPL", "MSFT"], group_by="ticker")
# Access: data["AAPL"]["Close"]

# Flatten for CSV round-tripping
data.to_csv("prices.csv")
# Read back with multi-level header:
df = pd.read_csv("prices.csv", header=[0, 1], index_col=0, parse_dates=True)

# Disable multi-level (recent versions)
data = yf.download(["AAPL", "MSFT"], multi_level_index=False)
```

---

## Project Integration

This project wraps yfinance through a dedicated client layer:

- @yfinance_client/client.py — Singleton `YFinanceClient` with LRU caching, rate limiting, circuit breaker, and retry logic
- @yfinance_client/cache.py — Thread-safe LRU cache (3000 entries)
- @yfinance_client/rate_limiter.py — Per-key rate limiter (0.1s delay)
- @yfinance_client/circuit_breaker.py — Exponential backoff circuit breaker (max 10 attempts)
- @yfinance_client/retry.py — Generic retry with backoff and validation callbacks
- @yfinance_client/protocols.py — Protocol interfaces for dependency injection
- @yfinance_client/news_client.py — News fetching with article enrichment
- @yfinance_client/article_scraper.py — HTML article content extraction (BeautifulSoup)
- @yfinance_client/news_aggregator.py — Country-level news aggregation with deduplication
- @optimizer/adapters/yfinance/price_adapter.py — PriceProvider adapter
- @optimizer/adapters/yfinance/market_cap_adapter.py — MarketCapProvider adapter

**Usage pattern:** Always use `YFinanceClient.get_instance()` or `get_yfinance_client()` — never call `yf.Ticker()` directly in application code.

---

## Key Constants

### Valid Periods

`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

### Valid Intervals

`1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

### Market Identifiers

`us_market`, `gb_market`, `de_market`, `fr_market`, `jp_market`, `hk_market`, `ca_market`, `in_market`, `br_market`, `au_market`

### Sector Keys

`technology`, `healthcare`, `financial-services`, `consumer-cyclical`, `communication-services`, `industrials`, `consumer-defensive`, `energy`, `basic-materials`, `real-estate`, `utilities`

### Screener Field Categories

- **Valuation:** `marketcap`, `peratio`, `pbratio`, `enterprisevalue`, `evtoebitda`, `evtorevenue`, `pricetosales`
- **Price:** `intradayprice`, `intradaymarketcap`, `fiftytwowkhigh`, `fiftytwoweeklow`, `intradaypricepctchange`
- **Dividends:** `dividendyield`, `trailingannualdividendyield`, `payoutratio`
- **Financials:** `revenue`, `ebitda`, `netincome`, `totaldebt`, `totalcash`, `grossprofitmargin`, `operatingmargin`, `profitmargin`
- **Growth:** `revenuegrowthquarterly`, `earningsgrowthquarterly`
- **Classification:** `sector`, `industry`, `exchange`, `region`
- **Analyst:** `recommendationkey`, `numberofanalystopinions`, `targetmeanprice`

## Implementation Patterns

See @.claude/skills/yfinance/PATTERNS.md for detailed code patterns.
