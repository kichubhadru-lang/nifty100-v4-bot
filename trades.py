from __future__ import annotations
import csv
from datetime import datetime
from typing import Any
import config
from data_provider import intraday
from utils import load_json, save_json, telegram_send


TRADE_FIELDS = [
    "trade_id", "symbol", "signal_date", "entry_date", "exit_date",
    "entry", "exit", "stop", "target1", "target2", "return_pct",
    "exit_reason", "status", "total_score", "fundamental_score",
    "technical_score", "market_score", "liquidity_score", "news_score",
]


def confirmation(candidate: dict[str, Any]) -> dict[str, Any] | None:
    df = intraday(candidate["symbol"])
    if len(df) < 4:
        return None
    first = df.iloc[:3]
    opening = float(first["Open"].iloc[0])
    high15 = float(first["High"].max())
    low15 = float(first["Low"].min())
    close15 = float(first["Close"].iloc[-1])
    current = float(df["Close"].iloc[-1])
    gap = (opening / candidate["close"] - 1) * 100
    bullish = close15 > opening
    valid = (
        config.MIN_GAP_PCT <= gap <= config.MAX_GAP_PCT
        and current > high15
        and (bullish or not config.REQUIRE_BULLISH_FIRST_15M)
    )
    return {
        "valid": valid,
        "open": round(opening, 2),
        "high15": round(high15, 2),
        "low15": round(low15, 2),
        "close15": round(close15, 2),
        "current": round(current, 2),
        "gap_pct": round(gap, 2),
        "bullish": bullish,
    }


def run_confirm() -> None:
    watchlist = load_json(config.WATCHLIST_FILE, {})
    candidates = watchlist.get("candidates", [])
    if not candidates:
        telegram_send("⏸️ V4: No saved candidates to confirm.")
        return

    valid = []
    for candidate in candidates:
        try:
            result = confirmation(candidate)
            if result and result["valid"]:
                valid.append((candidate, result))
        except Exception as exc:
            print(candidate["symbol"], exc)

    if not valid:
        telegram_send("⏸️ <b>V4: No entry</b>\nEvery candidate failed confirmation.")
        return

    # Hard rule: only the highest evening score among confirmed setups.
    candidate, result = sorted(valid, key=lambda x: x[0]["score"], reverse=True)[0]
    entry = result["current"]
    trade_id = f"{datetime.now():%Y%m%d}-{candidate['symbol'].replace('.NS','')}"
    trade = {
        "trade_id": trade_id,
        "symbol": candidate["symbol"],
        "signal_date": watchlist.get("date"),
        "entry_date": str(datetime.now().date()),
        "entry": entry,
        "stop": round(entry * (1 - config.STOP_LOSS_PCT / 100), 2),
        "target1": round(entry * (1 + config.TARGET_1_PCT / 100), 2),
        "target2": round(entry * (1 + config.TARGET_2_PCT / 100), 2),
        "status": "OPEN",
        "total_score": candidate["score"],
        **{f"{k}_score": v for k, v in candidate["component_scores"].items()},
    }
    open_trades = load_json(config.OPEN_TRADES_FILE, [])
    open_trades.append(trade)
    save_json(config.OPEN_TRADES_FILE, open_trades)

    telegram_send(
        "🚦 <b>V4 PAPER ENTRY</b>\n"
        f"Stock: {candidate['symbol'].replace('.NS','')}\n"
        f"Score: {candidate['score']}/100\n"
        f"Entry reference: ₹{trade['entry']}\n"
        f"Stop: ₹{trade['stop']}\n"
        f"Target 1: ₹{trade['target1']}\n"
        f"Target 2: ₹{trade['target2']}\n"
        "Maximum one trade. This is a paper signal, not an executed order."
    )


def append_trade_log(trade: dict[str, Any]) -> None:
    exists = config.TRADE_LOG_FILE.exists()
    with config.TRADE_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: trade.get(key, "") for key in TRADE_FIELDS})


def run_close() -> None:
    open_trades = load_json(config.OPEN_TRADES_FILE, [])
    remaining = []
    closed = []

    for trade in open_trades:
        try:
            df = intraday(trade["symbol"])
            if df.empty:
                remaining.append(trade)
                continue
            day_low = float(df["Low"].min())
            day_high = float(df["High"].max())
            last = float(df["Close"].iloc[-1])

            # Conservative ordering when both target and stop occur in available bars.
            if day_low <= trade["stop"] and day_high >= trade["target1"]:
                exit_price, reason = trade["stop"], "BOTH_HIT_ASSUME_STOP"
            elif day_low <= trade["stop"]:
                exit_price, reason = trade["stop"], "STOP"
            elif day_high >= trade["target2"]:
                exit_price, reason = trade["target2"], "TARGET_2"
            elif day_high >= trade["target1"]:
                exit_price, reason = trade["target1"], "TARGET_1"
            else:
                exit_price, reason = round(last, 2), "TIME_EXIT"

            trade.update({
                "exit_date": str(datetime.now().date()),
                "exit": exit_price,
                "return_pct": round((exit_price / trade["entry"] - 1) * 100, 3),
                "exit_reason": reason,
                "status": "CLOSED",
            })
            append_trade_log(trade)
            closed.append(trade)
        except Exception:
            remaining.append(trade)

    save_json(config.OPEN_TRADES_FILE, remaining)
    if not closed:
        telegram_send("V4 close check: no paper trade was closed.")
        return

    lines = ["🏁 <b>V4 PAPER OUTCOME</b>", ""]
    for t in closed:
        lines.extend([
            f"{t['symbol'].replace('.NS','')}: {t['return_pct']}%",
            f"Exit: ₹{t['exit']} ({t['exit_reason']})",
            "",
        ])
    telegram_send("\n".join(lines))
