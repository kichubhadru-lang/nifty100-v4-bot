from __future__ import annotations

import csv
import math
from datetime import datetime
from typing import Any

import config
from data_provider import intraday
from utils import load_json, save_json, telegram_send


TRADE_FIELDS = [
    "trade_id",
    "symbol",
    "signal_date",
    "entry_date",
    "exit_date",
    "entry",
    "exit",
    "stop",
    "target1",
    "target2",
    "quantity",
    "capital",
    "capital_used",
    "risk_budget",
    "actual_stop_risk",
    "estimated_cost",
    "gross_pnl",
    "net_pnl",
    "return_pct",
    "exit_reason",
    "status",
    "total_score",
    "fundamental_score",
    "technical_score",
    "market_score",
    "liquidity_score",
    "news_score",
]


def calculate_position_size(
    entry_price: float,
    stop_price: float,
    capital: float,
) -> dict[str, float | int]:
    """
    Calculate quantity using:
    - maximum risk per trade
    - maximum capital allocation
    """

    if entry_price <= 0 or stop_price <= 0 or capital <= 0:
        return {
            "quantity": 0,
            "risk_budget": 0.0,
            "capital_used": 0.0,
            "actual_stop_risk": 0.0,
            "estimated_cost": 0.0,
        }

    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        return {
            "quantity": 0,
            "risk_budget": 0.0,
            "capital_used": 0.0,
            "actual_stop_risk": 0.0,
            "estimated_cost": 0.0,
        }

    risk_budget = capital * (
        config.RISK_PER_TRADE_PCT / 100
    )

    quantity_by_risk = math.floor(
        risk_budget / risk_per_share
    )

    maximum_capital = capital * (
        config.MAX_CAPITAL_ALLOCATION_PCT / 100
    )

    quantity_by_capital = math.floor(
        maximum_capital / entry_price
    )

    quantity = min(
        quantity_by_risk,
        quantity_by_capital,
    )

    quantity = max(quantity, 0)

    capital_used = quantity * entry_price
    actual_stop_risk = quantity * risk_per_share

    estimated_cost = capital_used * (
        config.ESTIMATED_ROUND_TRIP_COST_PCT / 100
    )

    return {
        "quantity": quantity,
        "risk_budget": round(risk_budget, 2),
        "capital_used": round(capital_used, 2),
        "actual_stop_risk": round(actual_stop_risk, 2),
        "estimated_cost": round(estimated_cost, 2),
    }


def confirmation(
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    df = intraday(candidate["symbol"])

    if len(df) < 4:
        return None

    first = df.iloc[:3]

    opening = float(first["Open"].iloc[0])
    high15 = float(first["High"].max())
    low15 = float(first["Low"].min())
    close15 = float(first["Close"].iloc[-1])
    current = float(df["Close"].iloc[-1])

    gap = (
        opening / candidate["close"] - 1
    ) * 100

    bullish = close15 > opening
    breakout = current > high15

    valid = (
        config.MIN_GAP_PCT
        <= gap
        <= config.MAX_GAP_PCT

        and breakout

        and (
            bullish
            or not config.REQUIRE_BULLISH_FIRST_15M
        )
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
        "breakout": breakout,
    }


def run_confirm() -> None:
    watchlist = load_json(
        config.WATCHLIST_FILE,
        {},
    )

    candidates = watchlist.get(
        "candidates",
        [],
    )

    if not candidates:
        telegram_send(
            "⏸️ V4: No saved candidates to confirm."
        )
        return

    open_trades = load_json(
        config.OPEN_TRADES_FILE,
        [],
    )

    today = str(datetime.now().date())

    already_open_today = any(
        trade.get("entry_date") == today
        and trade.get("status") == "OPEN"
        for trade in open_trades
    )

    if already_open_today:
        telegram_send(
            "⏸️ <b>V4 confirmation skipped</b>\n"
            "Maximum one trade is already active today."
        )
        return

    valid: list[
        tuple[dict[str, Any], dict[str, Any]]
    ] = []

    for candidate in candidates:
        try:
            result = confirmation(candidate)

            if result and result["valid"]:
                valid.append(
                    (candidate, result)
                )

        except Exception as exc:
            print(
                candidate.get("symbol"),
                exc,
            )

    if not valid:
        telegram_send(
            "⏸️ <b>V4: No entry</b>\n"
            "Every candidate failed the "
            "15-minute confirmation."
        )
        return

    candidate, result = sorted(
        valid,
        key=lambda item: (
            item[0]["score"],
            item[0].get("volume_ratio", 0),
        ),
        reverse=True,
    )[0]

    entry = float(result["current"])

    stop = round(
        entry * (
            1 - config.STOP_LOSS_PCT / 100
        ),
        2,
    )

    target1 = round(
        entry * (
            1 + config.TARGET_1_PCT / 100
        ),
        2,
    )

    target2 = round(
        entry * (
            1 + config.TARGET_2_PCT / 100
        ),
        2,
    )

    sizing = calculate_position_size(
        entry_price=entry,
        stop_price=stop,
        capital=config.PAPER_CAPITAL,
    )

    quantity = int(sizing["quantity"])

    if quantity <= 0:
        telegram_send(
            "⚠️ <b>V4 entry rejected</b>\n"
            f"Stock: "
            f"{candidate['symbol'].replace('.NS', '')}\n"
            "Calculated quantity is zero."
        )
        return

    trade_id = (
        f"{datetime.now():%Y%m%d}-"
        f"{candidate['symbol'].replace('.NS', '')}"
    )

    trade = {
        "trade_id": trade_id,
        "symbol": candidate["symbol"],
        "signal_date": watchlist.get("date"),
        "entry_date": today,
        "entry": round(entry, 2),
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "quantity": quantity,
        "capital": config.PAPER_CAPITAL,
        "capital_used": sizing["capital_used"],
        "risk_budget": sizing["risk_budget"],
        "actual_stop_risk": sizing[
            "actual_stop_risk"
        ],
        "estimated_cost": sizing[
            "estimated_cost"
        ],
        "status": "OPEN",
        "total_score": candidate["score"],
        **{
            f"{key}_score": value
            for key, value
            in candidate[
                "component_scores"
            ].items()
        },
    }

    open_trades.append(trade)

    save_json(
        config.OPEN_TRADES_FILE,
        open_trades,
    )

    telegram_send(
        "🚦 <b>V4 PAPER ENTRY</b>\n\n"
        f"<b>Stock:</b> "
        f"{candidate['symbol'].replace('.NS', '')}\n"
        f"<b>Score:</b> "
        f"{candidate['score']}/100\n\n"
        f"<b>Entry:</b> ₹{trade['entry']:.2f}\n"
        f"<b>Stop:</b> ₹{trade['stop']:.2f}\n"
        f"<b>Target 1:</b> "
        f"₹{trade['target1']:.2f}\n"
        f"<b>Target 2:</b> "
        f"₹{trade['target2']:.2f}\n\n"
        f"<b>Quantity:</b> "
        f"{trade['quantity']} shares\n"
        f"<b>Paper capital:</b> "
        f"₹{trade['capital']:,.2f}\n"
        f"<b>Capital used:</b> "
        f"₹{trade['capital_used']:,.2f}\n"
        f"<b>Risk budget:</b> "
        f"₹{trade['risk_budget']:,.2f}\n"
        f"<b>Actual stop risk:</b> "
        f"₹{trade['actual_stop_risk']:,.2f}\n"
        f"<b>Estimated costs:</b> "
        f"₹{trade['estimated_cost']:,.2f}\n\n"
        f"<b>Gap:</b> "
        f"{result['gap_pct']}%\n"
        f"<b>First 15-minute high:</b> "
        f"₹{result['high15']:.2f}\n"
        f"<b>Bullish first candle:</b> "
        f"{result['bullish']}\n"
        f"<b>Breakout confirmed:</b> "
        f"{result['breakout']}\n\n"
        "Maximum one paper trade per day."
    )


def append_trade_log(
    trade: dict[str, Any],
) -> None:
    exists = config.TRADE_LOG_FILE.exists()

    config.TRADE_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config.TRADE_LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRADE_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                key: trade.get(key, "")
                for key in TRADE_FIELDS
            }
        )


def run_close() -> None:
    open_trades = load_json(
        config.OPEN_TRADES_FILE,
        [],
    )

    remaining: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []

    for trade in open_trades:
        try:
            df = intraday(trade["symbol"])

            if df.empty:
                remaining.append(trade)
                continue

            day_low = float(df["Low"].min())
            day_high = float(df["High"].max())
            last = float(df["Close"].iloc[-1])

            if (
                day_low <= trade["stop"]
                and day_high >= trade["target1"]
            ):
                exit_price = trade["stop"]
                reason = "BOTH_HIT_ASSUME_STOP"

            elif day_low <= trade["stop"]:
                exit_price = trade["stop"]
                reason = "STOP"

            elif day_high >= trade["target2"]:
                exit_price = trade["target2"]
                reason = "TARGET_2"

            elif day_high >= trade["target1"]:
                exit_price = trade["target1"]
                reason = "TARGET_1"

            else:
                exit_price = round(last, 2)
                reason = "TIME_EXIT"

            quantity = int(
                trade.get("quantity", 0)
            )

            entry = float(trade["entry"])

            gross_pnl = (
                exit_price - entry
            ) * quantity

            estimated_cost = float(
                trade.get(
                    "estimated_cost",
                    0.0,
                )
            )

            net_pnl = (
                gross_pnl
                - estimated_cost
            )

            capital_used = float(
                trade.get(
                    "capital_used",
                    entry * quantity,
                )
            )

            return_pct = (
                net_pnl
                / capital_used
                * 100
                if capital_used > 0
                else 0.0
            )

            trade.update(
                {
                    "exit_date": str(
                        datetime.now().date()
                    ),
                    "exit": round(
                        exit_price,
                        2,
                    ),
                    "gross_pnl": round(
                        gross_pnl,
                        2,
                    ),
                    "net_pnl": round(
                        net_pnl,
                        2,
                    ),
                    "return_pct": round(
                        return_pct,
                        3,
                    ),
                    "exit_reason": reason,
                    "status": "CLOSED",
                }
            )

            append_trade_log(trade)
            closed.append(trade)

        except Exception as exc:
            print(
                trade.get("symbol"),
                exc,
            )

            remaining.append(trade)

    save_json(
        config.OPEN_TRADES_FILE,
        remaining,
    )

    if not closed:
        telegram_send(
            "V4 close check: "
            "no paper trade was closed."
        )
        return

    lines = [
        "🏁 <b>V4 PAPER OUTCOME</b>",
        "",
    ]

    for trade in closed:
        result_icon = (
            "✅"
            if trade["net_pnl"] > 0
            else "❌"
        )

        lines.extend(
            [
                f"{result_icon} "
                f"<b>"
                f"{trade['symbol'].replace('.NS', '')}"
                f"</b>",
                f"Exit: ₹{trade['exit']:.2f}",
                f"Reason: {trade['exit_reason']}",
                f"Quantity: {trade['quantity']}",
                f"Gross P&L: "
                f"₹{trade['gross_pnl']:,.2f}",
                f"Costs: "
                f"₹{trade['estimated_cost']:,.2f}",
                f"Net P&L: "
                f"₹{trade['net_pnl']:,.2f}",
                f"Net return: "
                f"{trade['return_pct']}%",
                "",
            ]
        )

    telegram_send(
        "\n".join(lines)
    )
