from __future__ import annotations

import argparse
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from discord_utils import send_to_discord
from ep_detector import EPDetector
from ticker_utils import get_tickers


ROOT = Path(__file__).resolve().parent
DEFAULT_WATCHLIST = ROOT / "watchlist.csv"
DEFAULT_OUTPUT_DIR = ROOT / "daily_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily trading system using CANSLIM proxies, EP detection, and watchlist actions."
    )
    parser.add_argument("--watchlist-csv", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--ideas-limit", type=int, default=12)
    parser.add_argument("--watchlist-limit", type=int, default=20)
    parser.add_argument("--days", type=int, default=420)
    parser.add_argument("--use-cache", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--use-yfinance-only", action="store_true")
    return parser.parse_args()


def ensure_watchlist_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text("symbol,shares,cost_basis,notes\n", encoding="ascii")


def materialize_watchlist_from_env(path: Path) -> bool:
    raw_csv = os.getenv("WATCHLIST_CSV")
    raw_b64 = os.getenv("WATCHLIST_CSV_BASE64")

    if raw_b64:
        decoded = base64.b64decode(raw_b64).decode("utf-8")
        path.write_text(decoded, encoding="utf-8")
        return True

    if raw_csv:
        path.write_text(raw_csv, encoding="utf-8")
        return True

    return False


def load_watchlist(path: Path) -> pd.DataFrame:
    materialize_watchlist_from_env(path)
    ensure_watchlist_file(path)
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "shares", "cost_basis", "notes"])

    frame.columns = [column.strip().lower() for column in frame.columns]
    for column in ["symbol", "shares", "cost_basis", "notes"]:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame.loc[frame["symbol"].isin(["", "NAN", "NONE"]), "symbol"] = np.nan
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["cost_basis"] = pd.to_numeric(frame["cost_basis"], errors="coerce")
    return frame[["symbol", "shares", "cost_basis", "notes"]].dropna(subset=["symbol"])


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    clean = frame.copy()
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean.columns = [str(column).title() for column in clean.columns]
    expected = ["Open", "High", "Low", "Close", "Volume"]
    clean = clean[[column for column in expected if column in clean.columns]]
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0]
    return clean.sort_index()


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def download_yfinance_batch(
    tickers: list[str], days: int
) -> dict[str, pd.DataFrame]:
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=days)
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    histories: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values("Ticker"):
                continue
            histories[ticker] = normalize_history(data.xs(ticker, axis=1, level="Ticker"))
    elif tickers:
        histories[tickers[0]] = normalize_history(data)
    return {ticker: frame for ticker, frame in histories.items() if not frame.empty}


def build_price_cache(
    tickers: list[str],
    days: int,
    batch_size: int,
    use_yfinance_only: bool,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    use_alpaca = bool(alpaca_key and alpaca_secret and not use_yfinance_only)

    if use_alpaca:
        detector = EPDetector(api_key=alpaca_key, secret_key=alpaca_secret)
        for batch in chunked(tickers, batch_size):
            batch_data = detector.get_batch_stock_data(batch, days=days)
            for ticker, frame in batch_data.items():
                histories[ticker] = normalize_history(frame)
        return histories

    for batch in chunked(tickers, batch_size):
        histories.update(download_yfinance_batch(batch, days))
    return histories


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["Close"]
    high = enriched["High"]
    low = enriched["Low"]
    volume = enriched["Volume"]

    enriched["EMA10"] = close.ewm(span=10, adjust=False).mean()
    enriched["EMA21"] = close.ewm(span=21, adjust=False).mean()
    enriched["SMA50"] = close.rolling(50).mean()
    enriched["SMA200"] = close.rolling(200).mean()
    enriched["AVG_VOL20"] = volume.rolling(20).mean()
    enriched["AVG_DOLLAR_VOL20"] = (close * volume).rolling(20).mean()
    enriched["HIGH20"] = high.rolling(20).max()
    enriched["HIGH55"] = high.rolling(55).max()
    enriched["HIGH252"] = high.rolling(252).max()
    enriched["LOW20"] = low.rolling(20).min()
    enriched["LOW63"] = low.rolling(63).min()
    enriched["HIGH63"] = high.rolling(63).max()
    enriched["DRAWDOWN_63"] = close / enriched["HIGH63"] - 1.0
    enriched["CLOSE_VS_EMA21"] = close / enriched["EMA21"] - 1.0
    enriched["CLOSE_VS_SMA50"] = close / enriched["SMA50"] - 1.0
    enriched["SMA50_SLOPE20"] = enriched["SMA50"] / enriched["SMA50"].shift(20) - 1.0
    returns = close.pct_change()
    enriched["DIST_DAY"] = (
        (returns <= -0.03)
        & (volume > volume.shift(1))
        & (volume > enriched["AVG_VOL20"])
    )
    enriched["DIST_COUNT10"] = enriched["DIST_DAY"].rolling(10).sum().fillna(0.0)
    enriched["VOL_SURGE"] = volume / enriched["AVG_VOL20"]
    return enriched


def score_market_regime(benchmark_histories: dict[str, pd.DataFrame]) -> dict[str, Any]:
    states = []
    score = 0
    detail_lines = []

    for ticker in ["SPY", "QQQ"]:
        frame = add_indicators(benchmark_histories[ticker])
        latest = frame.iloc[-1]
        above_50 = bool(latest["Close"] > latest["SMA50"])
        above_200 = bool(latest["Close"] > latest["SMA200"])
        trend_up = bool(latest["SMA50"] > latest["SMA200"])
        instrument_score = int(above_50) + int(above_200) + int(trend_up)
        score += instrument_score
        states.append(instrument_score)
        detail_lines.append(
            f"{ticker} close ${latest['Close']:.2f} | above50={above_50} above200={above_200} trend_up={trend_up}"
        )

    if min(states) == 3:
        state = "BULLISH"
    elif score >= 3:
        state = "CAUTION"
    else:
        state = "RISK_OFF"

    return {
        "state": state,
        "score": score,
        "detail": detail_lines,
        "adds_allowed": state == "BULLISH",
    }


def compute_ep_signal(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 120:
        return {"is_ep": False, "ep_score": 0, "gap_percent": 0.0, "volume_surge": 0.0}

    current = frame.iloc[-1]
    previous_close = frame["Close"].iloc[-2]
    gap_percent = ((current["Open"] - previous_close) / previous_close) * 100 if previous_close else 0.0
    volume_surge = float(current["Volume"] / frame["Volume"].iloc[-21:-1].mean()) if len(frame) > 20 else 0.0
    consolidation = frame.iloc[-125:-5]
    range_percent = 999.0
    is_sideways = False
    if len(consolidation) >= 60:
        high = consolidation["High"].max()
        low = consolidation["Low"].min()
        range_percent = ((high - low) / low) * 100 if low else 999.0
        is_sideways = range_percent < 40

    price_above_50 = bool(current["Close"] > current.get("SMA50", np.nan))
    recent_high = bool(current["Close"] >= frame["High"].iloc[-20:].max())

    score = 0
    notes: list[str] = []
    if gap_percent >= 15:
        score += 40
        notes.append(f"gap {gap_percent:.1f}%")
    elif gap_percent >= 10:
        score += 30
        notes.append(f"gap {gap_percent:.1f}%")

    if volume_surge >= 5:
        score += 30
        notes.append(f"volume {volume_surge:.1f}x")
    elif volume_surge >= 3:
        score += 25
        notes.append(f"volume {volume_surge:.1f}x")
    elif volume_surge >= 2:
        score += 10
        notes.append(f"volume {volume_surge:.1f}x")

    if is_sideways and len(consolidation) >= 60:
        score += 20
        notes.append(f"sideways {len(consolidation)}d")
    elif len(consolidation) >= 30:
        score += 10

    if recent_high:
        score += 5
    if price_above_50:
        score += 5

    return {
        "is_ep": gap_percent >= 10.0 and score >= 30,
        "ep_score": score,
        "gap_percent": float(gap_percent),
        "volume_surge": float(volume_surge),
        "range_percent": float(range_percent),
        "notes": notes,
    }


def technical_prefilter(frame: pd.DataFrame) -> bool:
    enriched = add_indicators(frame)
    if len(enriched) < 252:
        return False

    latest = enriched.iloc[-1]
    ep_signal = compute_ep_signal(enriched)
    leader_trend = bool(
        latest["Close"] >= 5
        and latest["AVG_DOLLAR_VOL20"] >= 10_000_000
        and latest["Close"] > latest["SMA50"] > latest["SMA200"]
        and latest["SMA50_SLOPE20"] > 0
        and latest["Close"] >= latest["HIGH252"] * 0.85
    )
    breakout_zone = bool(latest["Close"] >= enriched["HIGH55"].shift(1).iloc[-1] * 0.97)
    return ep_signal["is_ep"] or (leader_trend and breakout_zone)


def relative_strength_vs_benchmark(
    frame: pd.DataFrame, benchmark: pd.DataFrame, lookback: int
) -> float:
    if len(frame) <= lookback or len(benchmark) <= lookback:
        return 0.0
    stock_return = frame["Close"].iloc[-1] / frame["Close"].iloc[-lookback - 1] - 1.0
    benchmark_return = benchmark["Close"].iloc[-1] / benchmark["Close"].iloc[-lookback - 1] - 1.0
    return float(stock_return - benchmark_return)


def info_value(info: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in info and info[key] is not None:
            return info[key]
    return None


def fetch_info_snapshot(ticker: str) -> dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def compute_canslim_proxy_score(
    ticker: str,
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    info: dict[str, Any],
    market_regime: dict[str, Any],
    ep_signal: dict[str, Any],
) -> dict[str, Any]:
    latest = frame.iloc[-1]
    score = 0
    notes: list[str] = []

    quarterly_growth = info_value(info, "earningsQuarterlyGrowth", "earningsGrowth")
    revenue_growth = info_value(info, "revenueGrowth")
    roe = info_value(info, "returnOnEquity")
    inst_owned = info_value(info, "heldPercentInstitutions", "institutionPercentHeld")
    float_shares = info_value(info, "floatShares", "sharesOutstanding")
    market_cap = info_value(info, "marketCap")

    if quarterly_growth is not None and quarterly_growth >= 0.25:
        score += 20
        notes.append(f"C {quarterly_growth:.0%} qtr growth")
    elif quarterly_growth is not None and quarterly_growth >= 0.10:
        score += 10
        notes.append(f"C {quarterly_growth:.0%} qtr growth")

    if revenue_growth is not None and revenue_growth >= 0.20:
        score += 15
        notes.append(f"A {revenue_growth:.0%} revenue growth")
    elif roe is not None and roe >= 0.15:
        score += 10
        notes.append(f"A {roe:.0%} ROE")

    if ep_signal["is_ep"]:
        score += 15
        notes.append(f"N EP {ep_signal['gap_percent']:.1f}% gap")
    elif latest["Close"] >= latest["HIGH252"] * 0.98:
        score += 10
        notes.append("N near 52w high")

    turnover = None
    if float_shares and float_shares > 0:
        turnover = latest["Volume"] / float_shares

    if latest["VOL_SURGE"] >= 1.5 and latest["AVG_DOLLAR_VOL20"] >= 10_000_000:
        score += 10
        notes.append(f"S vol {latest['VOL_SURGE']:.1f}x")
    if turnover is not None and turnover >= 0.005:
        score += 5
        notes.append(f"S turnover {turnover:.2%}")

    rs_3m = relative_strength_vs_benchmark(frame, benchmark, 63)
    rs_6m = relative_strength_vs_benchmark(frame, benchmark, 126)
    if rs_3m >= 0.10 and rs_6m >= 0.15:
        score += 15
        notes.append(f"L RS +{rs_6m:.0%} vs SPY")
    elif rs_3m >= 0.05:
        score += 8
        notes.append(f"L RS +{rs_3m:.0%} vs SPY")

    if inst_owned is not None and 0.20 <= inst_owned <= 0.95:
        score += 10
        notes.append(f"I {inst_owned:.0%} institutions")
    elif inst_owned is not None and inst_owned >= 0.10:
        score += 5
        notes.append(f"I {inst_owned:.0%} institutions")

    if market_regime["state"] == "BULLISH":
        score += 10
        notes.append("M bullish tape")
    elif market_regime["state"] == "CAUTION":
        score += 5
        notes.append("M mixed tape")

    breakout_ready = bool(
        latest["Close"] >= frame["HIGH55"].shift(1).iloc[-1] * 0.995
        and latest["VOL_SURGE"] >= 1.2
        and latest["Close"] > latest["SMA50"] > latest["SMA200"]
    )

    if market_cap is not None and market_cap < 1_000_000_000:
        notes.append("sub-$1B cap")

    return {
        "ticker": ticker,
        "canslim_score": int(score),
        "notes": notes,
        "rs_3m": rs_3m,
        "rs_6m": rs_6m,
        "quarterly_growth": quarterly_growth,
        "revenue_growth": revenue_growth,
        "institutional_ownership": inst_owned,
        "market_cap": market_cap,
        "breakout_ready": breakout_ready,
    }


def classify_new_idea(
    ticker: str,
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_regime: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any] | None:
    enriched = add_indicators(frame)
    latest = enriched.iloc[-1]

    if len(enriched) < 252:
        return None
    if latest["Close"] < 5 or latest["AVG_DOLLAR_VOL20"] < 10_000_000:
        return None

    ep_signal = compute_ep_signal(enriched)
    canslim = compute_canslim_proxy_score(
        ticker=ticker,
        frame=enriched,
        benchmark=benchmark,
        info=info,
        market_regime=market_regime,
        ep_signal=ep_signal,
    )

    leader_trend = bool(
        latest["Close"] > latest["SMA50"] > latest["SMA200"]
        and latest["SMA50_SLOPE20"] > 0
        and latest["Close"] >= latest["HIGH252"] * 0.85
    )
    if not leader_trend and not ep_signal["is_ep"]:
        return None

    total_score = canslim["canslim_score"] + min(ep_signal["ep_score"], 20)
    prior_high55 = enriched["HIGH55"].shift(1).iloc[-1]
    trigger_price = max(prior_high55, latest["High"]) * 1.001
    stop_price = min(latest["EMA21"], latest["SMA50"]) * 0.98

    if total_score >= 85 and market_regime["state"] == "BULLISH" and canslim["breakout_ready"]:
        action = "BUY_PILOT"
    elif total_score >= 70 and market_regime["state"] != "RISK_OFF":
        action = "WATCH_FOR_TRIGGER"
    else:
        return None

    reasons = []
    if ep_signal["is_ep"]:
        reasons.append(f"EP {ep_signal['gap_percent']:.1f}% gap / {ep_signal['volume_surge']:.1f}x vol")
    reasons.extend(canslim["notes"][:4])

    return {
        "ticker": ticker,
        "action": action,
        "total_score": int(total_score),
        "canslim_score": canslim["canslim_score"],
        "ep_score": ep_signal["ep_score"],
        "close": float(latest["Close"]),
        "trigger_price": float(trigger_price),
        "stop_price": float(stop_price),
        "market_cap": canslim["market_cap"],
        "reasons": reasons,
    }


def classify_watchlist_position(
    ticker: str,
    frame: pd.DataFrame,
    market_regime: dict[str, Any],
    cost_basis: float | None,
    shares: float | None,
) -> dict[str, Any]:
    enriched = add_indicators(frame)
    latest = enriched.iloc[-1]
    previous = enriched.iloc[-2]

    close = float(latest["Close"])
    pnl_pct = (close / cost_basis - 1.0) if cost_basis and cost_basis > 0 else np.nan

    strong_trend = bool(
        latest["Close"] > latest["SMA50"] > latest["SMA200"]
        and latest["SMA50_SLOPE20"] > 0
        and latest["AVG_DOLLAR_VOL20"] >= 10_000_000
    )
    breakout_ready = bool(
        latest["Close"] > enriched["HIGH20"].shift(1).iloc[-1]
        and latest["VOL_SURGE"] >= 1.25
        and latest["CLOSE_VS_SMA50"] <= 0.15
    )
    pullback_add = bool(
        strong_trend
        and -0.18 <= latest["DRAWDOWN_63"] <= -0.06
        and (latest["Low"] <= latest["EMA21"] * 1.02 or latest["Low"] <= latest["SMA50"] * 1.02)
        and latest["Close"] > latest["EMA10"]
        and latest["Close"] >= previous["Close"]
    )
    extended = bool(latest["CLOSE_VS_SMA50"] > 0.25 or latest["CLOSE_VS_EMA21"] > 0.15)
    first_50d_break = bool(latest["Close"] < latest["SMA50"] and previous["Close"] >= previous["SMA50"])
    hedge_signal = bool(
        (latest["Close"] < latest["SMA50"] and latest["DIST_COUNT10"] >= 2)
        or (market_regime["state"] == "RISK_OFF" and latest["Close"] < latest["SMA50"])
    )
    exit_signal = bool(
        latest["Close"] < latest["SMA200"]
        and previous["Close"] < previous["SMA200"]
        and latest["DRAWDOWN_63"] <= -0.20
    )

    if exit_signal:
        action = "EXIT"
        suggestion = f"Exit the remaining position. Long-term trend is broken. Hard line: ${latest['SMA200']:.2f}."
    elif hedge_signal:
        action = "HEDGE"
        suggestion = (
            f"Hedge or trim 25-50% of the position. Price lost the 50d trend. "
            f"Keep a hard stop near ${latest['SMA200'] * 0.99:.2f}."
        )
    elif strong_trend and breakout_ready and (np.isnan(pnl_pct) or pnl_pct >= 0.03) and market_regime["adds_allowed"]:
        action = "ADD_ON_BREAKOUT"
        suggestion = (
            f"Add 20-25% only above ${enriched['HIGH20'].shift(1).iloc[-1] * 1.001:.2f}. "
            f"Initial stop near ${min(latest['EMA21'], latest['SMA50']) * 0.98:.2f}."
        )
    elif pullback_add and (np.isnan(pnl_pct) or pnl_pct >= 0.03) and market_regime["state"] != "RISK_OFF":
        action = "ADD_ON_PULLBACK"
        suggestion = (
            f"Add 10-15% on this rebound if the stock keeps holding the 21d/50d area. "
            f"Stop near ${min(latest['EMA21'], latest['SMA50']) * 0.98:.2f}."
        )
    elif extended or first_50d_break or latest["DIST_COUNT10"] >= 3:
        action = "PREPARE_HEDGE"
        suggestion = (
            f"No new adds. Tighten the stop to about ${latest['SMA50'] * 0.98:.2f} "
            f"and consider a 10-25% trim or partial hedge."
        )
    else:
        action = "HOLD"
        suggestion = "Hold. No add or hedge trigger today."

    return {
        "ticker": ticker,
        "action": action,
        "close": close,
        "shares": shares,
        "cost_basis": cost_basis,
        "pnl_pct": pnl_pct,
        "suggestion": suggestion,
        "drawdown_63": float(latest["DRAWDOWN_63"]),
        "dist_count10": float(latest["DIST_COUNT10"]),
    }


def format_market_cap(market_cap: Any) -> str:
    if market_cap is None or pd.isna(market_cap):
        return "n/a"
    market_cap = float(market_cap)
    if market_cap >= 1_000_000_000_000:
        return f"{market_cap / 1_000_000_000_000:.1f}T"
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.1f}B"
    if market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.1f}M"
    return f"{market_cap:,.0f}"


def build_report(
    generated_at: datetime,
    market_regime: dict[str, Any],
    ideas: list[dict[str, Any]],
    watchlist_actions: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Daily Trading System Report | {generated_at.strftime('%Y-%m-%d')}",
        "",
        f"Market regime: **{market_regime['state']}** (score {market_regime['score']}/6)",
    ]
    lines.extend(f"- {detail}" for detail in market_regime["detail"])
    lines.append("")
    lines.append("## New Ideas")
    if not ideas:
        lines.append("- No new CANSLIM/EP ideas passed today's thresholds.")
    else:
        for idea in ideas:
            reason_text = "; ".join(idea["reasons"][:4])
            lines.append(
                f"- {idea['ticker']} | {idea['action']} | score {idea['total_score']} "
                f"(CANSLIM {idea['canslim_score']}, EP {idea['ep_score']}) | "
                f"close ${idea['close']:.2f} | trigger ${idea['trigger_price']:.2f} | "
                f"stop ${idea['stop_price']:.2f} | cap {format_market_cap(idea['market_cap'])}"
            )
            lines.append(f"  Why: {reason_text}")

    lines.append("")
    lines.append("## Watchlist")
    if not watchlist_actions:
        lines.append("- Watchlist is empty. Fill in watchlist.csv to receive holding-specific actions.")
    else:
        for item in watchlist_actions:
            pnl_text = "n/a" if pd.isna(item["pnl_pct"]) else f"{item['pnl_pct']:.1%}"
            lines.append(
                f"- {item['ticker']} | {item['action']} | close ${item['close']:.2f} | P/L {pnl_text}"
            )
            lines.append(f"  Plan: {item['suggestion']}")

    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    watchlist = load_watchlist(args.watchlist_csv)
    generated_at = datetime.utcnow()

    benchmark_histories = build_price_cache(["SPY", "QQQ"], days=320, batch_size=2, use_yfinance_only=True)
    market_regime = score_market_regime(benchmark_histories)

    universe = get_tickers(use_cache=args.use_cache)
    if args.max_tickers:
        universe = universe[: args.max_tickers]

    universe_histories = build_price_cache(
        tickers=universe,
        days=args.days,
        batch_size=args.batch_size,
        use_yfinance_only=args.use_yfinance_only,
    )

    ideas: list[dict[str, Any]] = []
    for ticker, history in universe_histories.items():
        try:
            if not technical_prefilter(history):
                continue
            info = fetch_info_snapshot(ticker)
            candidate = classify_new_idea(
                ticker=ticker,
                frame=history,
                benchmark=benchmark_histories["SPY"],
                market_regime=market_regime,
                info=info,
            )
            if candidate is not None:
                ideas.append(candidate)
        except Exception:
            continue

    ideas = sorted(ideas, key=lambda item: item["total_score"], reverse=True)[: args.ideas_limit]

    watchlist_symbols = [symbol for symbol in watchlist["symbol"].tolist() if symbol]
    watchlist_histories = {
        ticker: universe_histories[ticker]
        for ticker in watchlist_symbols
        if ticker in universe_histories
    }
    missing_watchlist = [ticker for ticker in watchlist_symbols if ticker not in watchlist_histories]
    if missing_watchlist:
        watchlist_histories.update(
            build_price_cache(
                tickers=missing_watchlist,
                days=args.days,
                batch_size=max(1, min(10, len(missing_watchlist))),
                use_yfinance_only=args.use_yfinance_only,
            )
        )

    watchlist_actions: list[dict[str, Any]] = []
    for _, row in watchlist.iterrows():
        ticker = row["symbol"]
        history = watchlist_histories.get(ticker)
        if history is None or history.empty:
            continue
        watchlist_actions.append(
            classify_watchlist_position(
                ticker=ticker,
                frame=history,
                market_regime=market_regime,
                cost_basis=row["cost_basis"] if pd.notna(row["cost_basis"]) else None,
                shares=row["shares"] if pd.notna(row["shares"]) else None,
            )
        )

    action_priority = {
        "EXIT": 0,
        "HEDGE": 1,
        "PREPARE_HEDGE": 2,
        "ADD_ON_BREAKOUT": 3,
        "ADD_ON_PULLBACK": 4,
        "HOLD": 5,
    }
    watchlist_actions = sorted(
        watchlist_actions,
        key=lambda item: (action_priority.get(item["action"], 99), item["ticker"]),
    )[: args.watchlist_limit]

    report = build_report(
        generated_at=generated_at,
        market_regime=market_regime,
        ideas=ideas,
        watchlist_actions=watchlist_actions,
    )

    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"daily_trading_report_{timestamp}.md"
    latest_path = args.output_dir / "daily_trading_report_latest.md"
    report_path.write_text(report, encoding="ascii", errors="ignore")
    latest_path.write_text(report, encoding="ascii", errors="ignore")

    if args.no_discord or args.dry_run:
        print(report)
        print(f"\nReport written to {report_path}")
        return

    send_to_discord(
        content=report,
        title=f"Daily Trading System | {generated_at.strftime('%Y-%m-%d')}",
    )
    print(f"Report sent to Discord and written to {report_path}")


if __name__ == "__main__":
    main()
