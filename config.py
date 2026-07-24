from pathlib import Path

NIFTY100_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
BENCHMARK = "^NSEI"
VIX = "^INDIAVIX"

DATA_DIR = Path("data")
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
OPEN_TRADES_FILE = DATA_DIR / "open_trades.json"
TRADE_LOG_FILE = DATA_DIR / "trade_log.csv"
WEIGHTS_FILE = DATA_DIR / "adaptive_weights.json"

# Candidate rules
MIN_DROP_PCT = -6.0
MAX_DROP_PCT = -2.5
MIN_PRICE = 100.0
MIN_AVG_VALUE_20D = 500_000_000       # ₹50 crore
MIN_MARKET_CAP = 200_000_000_000      # ₹20,000 crore
MIN_ROE = 0.15
MAX_DEBT_TO_EQUITY_YAHOO = 50.0       # Yahoo convention: 50 ≈ 0.50
MIN_PROFIT_MARGIN = 0.0
MIN_DISTANCE_200DMA = -5.0
MAX_DISTANCE_200DMA = 20.0
MIN_RSI = 35.0
MAX_RSI = 52.0
MIN_VOLUME_RATIO = 1.20
MAX_CANDIDATES = 5
MIN_SIGNAL_SCORE = 80.0

# Market rules
MAX_NIFTY_FALL_PCT = -1.0
REQUIRE_NIFTY_ABOVE_50DMA = True
MAX_VIX = 20.0

# Next-day confirmation
CONFIRM_AFTER_MINUTES = 15
MIN_GAP_PCT = -1.5
MAX_GAP_PCT = 2.0
REQUIRE_BULLISH_FIRST_15M = True
MAX_TRADES_PER_DAY = 1

# Trade management
TARGET_1_PCT = 2.0
TARGET_2_PCT = 3.0
STOP_LOSS_PCT = 1.25
RISK_PER_TRADE_PCT = 0.50
TIME_EXIT_HOUR = 15
TIME_EXIT_MINUTE = 15

# Adaptive scoring
BASE_WEIGHTS = {
    "fundamental": 30.0,
    "technical": 30.0,
    "market": 20.0,
    "liquidity": 10.0,
    "news": 10.0,
}
MIN_COMPLETED_TRADES_TO_LEARN = 30
LEARNING_WINDOW = 100
MAX_WEIGHT_DEVIATION_PCT = 20.0
LEARNING_RATE = 0.15
MIN_BUCKET_TRADES = 8

# Conservative blocked-news terms. This is only a guardrail, not full news analysis.
BLOCKED_NEWS_TERMS = {
    "fraud", "sebi action", "investigation", "default", "bankruptcy",
    "insolvency", "downgrade", "promoter pledge", "resignation",
    "accounting irregularity", "raid", "penalty", "result miss",
    "weak guidance", "block deal", "stake sale",
}
