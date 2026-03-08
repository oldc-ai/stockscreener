from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "ten_x_stocks_with_fundamentals_filtered.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study clean 10-bagger bull runs and derive add/caution/exit rules."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV,
        help="Candidate universe exported by the existing 10x finder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for research outputs.",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="History start date passed to yfinance.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional history end date passed to yfinance.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of symbols to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Ticker batch size for yfinance downloads.",
    )
    parser.add_argument(
        "--min-factor",
        type=float,
        default=10.0,
        help="Minimum run multiple required after recomputing adjusted history.",
    )
    parser.add_argument(
        "--min-start-price",
        type=float,
        default=2.0,
        help="Minimum adjusted close at the bull-run trough.",
    )
    parser.add_argument(
        "--min-run-days",
        type=int,
        default=252,
        help="Minimum bull-run duration in trading days.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=756,
        help="Minimum available daily bars for a symbol to be considered clean.",
    )
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=5_000_000.0,
        help="Minimum median dollar volume during the run.",
    )
    parser.add_argument(
        "--min-early-dollar-volume",
        type=float,
        default=1_500_000.0,
        help="Minimum median dollar volume in the first 63 trading days of the run.",
    )
    parser.add_argument(
        "--max-p99-daily-move",
        type=float,
        default=0.35,
        help="Reject names whose 99th percentile absolute daily move exceeds this threshold.",
    )
    return parser.parse_args()


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_candidates(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["symbol"]).copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame = frame.sort_values(["marketCap", "max_return_factor"], ascending=[False, False])
    if limit is not None:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    clean = frame.copy()
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean.columns = [str(column).lower() for column in clean.columns]
    expected = ["open", "high", "low", "close", "volume"]
    clean = clean[[column for column in expected if column in clean.columns]]
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["close"])
    clean = clean[clean["close"] > 0]
    return clean.sort_index()


def download_histories(
    symbols: list[str], start: str, end: str | None, batch_size: int
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    histories: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for batch in chunked(symbols, batch_size):
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in batch:
                if symbol not in data.columns.get_level_values("Ticker"):
                    failed.append(symbol)
                    continue
                symbol_frame = data.xs(symbol, axis=1, level="Ticker")
                symbol_frame = normalize_history(symbol_frame)
                if symbol_frame.empty:
                    failed.append(symbol)
                    continue
                histories[symbol] = symbol_frame
        else:
            symbol = batch[0]
            single = normalize_history(data)
            if single.empty:
                failed.append(symbol)
            else:
                histories[symbol] = single

    return histories, sorted(set(failed))


def find_best_run(close: pd.Series) -> dict[str, object] | None:
    close = close.dropna()
    if close.empty:
        return None

    best_factor = 0.0
    best_start = None
    best_end = None
    low_price = np.inf
    low_date = None

    for date, price in close.items():
        if price <= 0:
            continue
        if price < low_price:
            low_price = float(price)
            low_date = date
        factor = float(price) / low_price
        if factor > best_factor:
            best_factor = factor
            best_start = low_date
            best_end = date

    if best_start is None or best_end is None:
        return None

    run_slice = close.loc[best_start:best_end]
    return {
        "max_return_factor": best_factor,
        "run_start_date": pd.Timestamp(best_start),
        "run_end_date": pd.Timestamp(best_end),
        "start_price": float(close.loc[best_start]),
        "end_price": float(close.loc[best_end]),
        "run_days": int(len(run_slice)),
    }


def assess_candidate(
    symbol: str, history: pd.DataFrame, args: argparse.Namespace
) -> dict[str, object] | None:
    run = find_best_run(history["close"])
    if run is None:
        return None

    run_slice = history.loc[run["run_start_date"] : run["run_end_date"]]
    early_run = run_slice.head(63)
    returns = run_slice["close"].pct_change().abs().dropna()

    run_dollar_volume = run_slice["close"] * run_slice["volume"]
    early_dollar_volume = early_run["close"] * early_run["volume"]

    result = {
        "symbol": symbol,
        **run,
        "history_days": int(len(history)),
        "median_dollar_volume_run": float(run_dollar_volume.median()) if not run_dollar_volume.empty else np.nan,
        "median_dollar_volume_early": float(early_dollar_volume.median()) if not early_dollar_volume.empty else np.nan,
        "p99_abs_daily_move": float(returns.quantile(0.99)) if not returns.empty else np.nan,
        "last_date": pd.Timestamp(history.index.max()),
    }

    fail_reasons: list[str] = []
    if result["max_return_factor"] < args.min_factor:
        fail_reasons.append("factor")
    if result["start_price"] < args.min_start_price:
        fail_reasons.append("start_price")
    if result["run_days"] < args.min_run_days:
        fail_reasons.append("run_days")
    if result["history_days"] < args.min_history_days:
        fail_reasons.append("history_days")
    if result["median_dollar_volume_run"] < args.min_dollar_volume:
        fail_reasons.append("run_liquidity")
    if result["median_dollar_volume_early"] < args.min_early_dollar_volume:
        fail_reasons.append("early_liquidity")
    if result["p99_abs_daily_move"] > args.max_p99_daily_move:
        fail_reasons.append("volatility")

    result["quality_pass"] = not fail_reasons
    result["fail_reasons"] = ",".join(fail_reasons)
    return result


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["close"]
    high = enriched["high"]
    low = enriched["low"]
    volume = enriched["volume"]

    enriched["ema10"] = close.ewm(span=10, adjust=False).mean()
    enriched["ema21"] = close.ewm(span=21, adjust=False).mean()
    enriched["sma50"] = close.rolling(50).mean()
    enriched["sma200"] = close.rolling(200).mean()
    enriched["avg_volume20"] = volume.rolling(20).mean()
    enriched["avg_dollar_volume20"] = (close * volume).rolling(20).mean()
    enriched["high20"] = close.rolling(20).max()
    enriched["high63"] = close.rolling(63).max()
    enriched["low20"] = close.rolling(20).min()
    enriched["low63"] = close.rolling(63).min()
    enriched["close_vs_ema21"] = close / enriched["ema21"] - 1.0
    enriched["close_vs_sma50"] = close / enriched["sma50"] - 1.0
    enriched["drawdown_63"] = close / enriched["high63"] - 1.0
    enriched["sma50_slope20"] = enriched["sma50"] / enriched["sma50"].shift(20) - 1.0
    enriched["distribution_day"] = (
        (close.pct_change() <= -0.03)
        & (volume > volume.shift(1))
        & (volume > enriched["avg_volume20"])
    )
    enriched["distribution_count10"] = (
        enriched["distribution_day"].rolling(10).sum().fillna(0.0)
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched["atr14"] = true_range.rolling(14).mean()
    return enriched


def debounce(signal: pd.Series, cooldown: int) -> pd.Series:
    active = 0
    output = pd.Series(False, index=signal.index)
    for index, flag in signal.fillna(False).items():
        if active > 0:
            active -= 1
        if flag and active == 0:
            output.loc[index] = True
            active = cooldown
    return output


def future_stats(close: pd.Series, date: pd.Timestamp, window: int) -> dict[str, float]:
    current_pos = close.index.get_indexer([date])[0]
    future = close.iloc[current_pos + 1 : current_pos + 1 + window]
    prefix = f"{window}d"

    if future.empty:
        return {
            f"return_{prefix}": np.nan,
            f"max_gain_{prefix}": np.nan,
            f"max_drawdown_{prefix}": np.nan,
        }

    entry_price = float(close.iloc[current_pos])
    return {
        f"return_{prefix}": float(future.iloc[-1] / entry_price - 1.0),
        f"max_gain_{prefix}": float(future.max() / entry_price - 1.0),
        f"max_drawdown_{prefix}": float(future.min() / entry_price - 1.0),
    }


def build_signal_table(
    symbol: str,
    history: pd.DataFrame,
    candidate: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    frame = add_indicators(history)

    trend_ready = (
        frame["sma200"].notna()
        & (frame["sma50"] > frame["sma200"])
        & (frame["sma50_slope20"] > 0)
        & (frame["close"] > frame["sma50"])
        & (frame["avg_dollar_volume20"] >= args.min_dollar_volume)
    )
    touch_support = (frame["low"] <= frame["ema21"] * 1.02) | (
        frame["low"] <= frame["sma50"] * 1.02
    )

    add_on_pullback = (
        trend_ready
        & frame["drawdown_63"].between(-0.18, -0.06)
        & touch_support
        & (frame["close"] > frame["ema10"])
        & (frame["close"] >= frame["close"].shift(1))
        & (frame["close_vs_sma50"] >= -0.01)
    )
    add_on_breakout = (
        trend_ready
        & (frame["close"] > frame["high20"].shift(1))
        & (frame["volume"] > frame["avg_volume20"] * 1.25)
        & frame["close_vs_sma50"].between(0.0, 0.15)
        & (frame["drawdown_63"] > -0.05)
    )
    prepare_for_drawback = (
        trend_ready
        & (
            (frame["close_vs_sma50"] > 0.25)
            | (frame["close_vs_ema21"] > 0.15)
            | (frame["distribution_count10"] >= 3)
            | (
                (frame["close"] < frame["sma50"])
                & (frame["close"].shift(1) >= frame["sma50"].shift(1))
                & (frame["distribution_count10"] >= 2)
            )
            | (
                (frame["close"] < frame["ema21"])
                & (frame["close"].shift(1) >= frame["ema21"].shift(1))
                & (frame["close_vs_sma50"] > 0.08)
            )
        )
    )
    exit_signal = (
        (frame["close"] < frame["sma200"])
        & (frame["close"].shift(1) < frame["sma200"].shift(1))
        & (frame["drawdown_63"] <= -0.20)
    )

    run_mask = (frame.index >= candidate["run_start_date"]) & (
        frame.index <= candidate["run_end_date"]
    )
    exit_mask = (frame.index >= candidate["run_start_date"]) & (
        frame.index <= min(
            candidate["run_end_date"] + pd.Timedelta(days=120),
            candidate["last_date"],
        )
    )

    signal_map = {
        "ADD_ON_PULLBACK": debounce(add_on_pullback & run_mask, cooldown=12),
        "ADD_ON_BREAKOUT": debounce(add_on_breakout & run_mask, cooldown=15),
        "PREPARE_FOR_DRAWBACK": debounce(
            prepare_for_drawback & run_mask, cooldown=10
        ),
        "EXIT": debounce(exit_signal & exit_mask, cooldown=20),
    }

    rows: list[dict[str, object]] = []
    for signal_name, signal in signal_map.items():
        for date in frame.index[signal]:
            event = {
                "symbol": symbol,
                "signal": signal_name,
                "date": pd.Timestamp(date),
                "close": float(frame.at[date, "close"]),
                "ema21_gap": float(frame.at[date, "close_vs_ema21"]),
                "sma50_gap": float(frame.at[date, "close_vs_sma50"]),
                "drawdown_63": float(frame.at[date, "drawdown_63"]),
                "distribution_count10": float(frame.at[date, "distribution_count10"]),
                "volume_multiple_20d": float(
                    frame.at[date, "volume"] / frame.at[date, "avg_volume20"]
                )
                if pd.notna(frame.at[date, "avg_volume20"]) and frame.at[date, "avg_volume20"] > 0
                else np.nan,
                **candidate,
            }
            event.update(future_stats(frame["close"], date, 20))
            event.update(future_stats(frame["close"], date, 60))
            rows.append(event)

    latest = frame.iloc[-1]
    latest_action = "HOLD"
    if bool(signal_map["EXIT"].iloc[-1]):
        latest_action = "EXIT"
    elif bool(signal_map["PREPARE_FOR_DRAWBACK"].iloc[-1]):
        latest_action = "PREPARE_FOR_DRAWBACK"
    elif bool(signal_map["ADD_ON_PULLBACK"].iloc[-1]):
        latest_action = "ADD_ON_PULLBACK"
    elif bool(signal_map["ADD_ON_BREAKOUT"].iloc[-1]):
        latest_action = "ADD_ON_BREAKOUT"

    latest_snapshot = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "latest_date": frame.index[-1],
                "latest_close": float(latest["close"]),
                "latest_action": latest_action,
                "close_vs_ema21": float(latest["close_vs_ema21"]),
                "close_vs_sma50": float(latest["close_vs_sma50"]),
                "drawdown_63": float(latest["drawdown_63"]),
                "distribution_count10": float(latest["distribution_count10"]),
                "run_start_date": candidate["run_start_date"],
                "run_end_date": candidate["run_end_date"],
                "max_return_factor": candidate["max_return_factor"],
            }
        ]
    )

    return pd.DataFrame(rows), latest_snapshot


def summarize_signals(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    grouped = events.groupby("signal", dropna=False)
    summary = grouped.agg(
        signals=("signal", "size"),
        median_return_20d=("return_20d", "median"),
        median_return_60d=("return_60d", "median"),
        median_max_gain_20d=("max_gain_20d", "median"),
        median_max_gain_60d=("max_gain_60d", "median"),
        median_max_drawdown_20d=("max_drawdown_20d", "median"),
        median_max_drawdown_60d=("max_drawdown_60d", "median"),
        win_rate_20d=("return_20d", lambda s: float((s > 0).mean())),
        win_rate_60d=("return_60d", lambda s: float((s > 0).mean())),
    )
    return summary.reset_index().sort_values("signal")


def format_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame.empty:
        return "No rows."
    preview = frame.head(max_rows).copy()
    return preview.to_string(index=False)


def signal_lookup(summary: pd.DataFrame, signal: str) -> pd.Series | None:
    if summary.empty:
        return None
    matches = summary.loc[summary["signal"] == signal]
    if matches.empty:
        return None
    return matches.iloc[0]


def write_report(
    output_dir: Path,
    candidates: pd.DataFrame,
    clean_candidates: pd.DataFrame,
    failed_downloads: list[str],
    signal_summary: pd.DataFrame,
    events: pd.DataFrame,
    latest_actions: pd.DataFrame,
) -> None:
    report_path = output_dir / "bull_run_report.md"

    top_clean = clean_candidates.sort_values(
        ["max_return_factor", "median_dollar_volume_run"], ascending=[False, False]
    )[
        [
            "symbol",
            "max_return_factor",
            "run_start_date",
            "run_end_date",
            "run_days",
            "start_price",
            "end_price",
            "median_dollar_volume_run",
        ]
    ]

    actionable_rules = pd.DataFrame(
        [
            {
                "action": "ADD_ON_BREAKOUT",
                "rule": "Price in strong uptrend, clears 20-day high on >=1.25x volume, still within 15% of 50-day average.",
            },
            {
                "action": "ADD_ON_PULLBACK",
                "rule": "Price pulls back 6-18% from a 63-day high, tests 21/50-day support, then closes back above the 10-day average.",
            },
            {
                "action": "PREPARE_FOR_DRAWBACK",
                "rule": "Price is extended (>25% above 50-day or >15% above 21-day), clusters 3 distribution days, or loses the 50-day average. Stop adding and tighten risk.",
            },
            {
                "action": "EXIT",
                "rule": "Full exit only after a real long-term trend break: two closes below the 200-day average with at least 20% damage off the 63-day high.",
            },
        ]
    )

    with report_path.open("w", encoding="ascii", errors="ignore") as handle:
        handle.write("# Bull Run System Exploration\n\n")
        handle.write("## Universe\n\n")
        handle.write(f"- Input candidates: {len(candidates)}\n")
        handle.write(f"- Clean 10-baggers after adjusted-data filters: {len(clean_candidates)}\n")
        handle.write(f"- Failed downloads: {len(failed_downloads)}\n\n")

        if failed_downloads:
            handle.write(
                "Failed download symbols: "
                + ", ".join(failed_downloads[:30])
                + ("\n\n" if len(failed_downloads) <= 30 else ", ...\n\n")
            )

        handle.write("## Action Rules\n\n")
        handle.write("```\n")
        handle.write(format_table(actionable_rules, max_rows=len(actionable_rules)))
        handle.write("\n```\n\n")

        handle.write("## Signal Summary\n\n")
        handle.write("```\n")
        handle.write(format_table(signal_summary, max_rows=len(signal_summary)))
        handle.write("\n```\n\n")

        breakout = signal_lookup(signal_summary, "ADD_ON_BREAKOUT")
        pullback = signal_lookup(signal_summary, "ADD_ON_PULLBACK")
        prepare = signal_lookup(signal_summary, "PREPARE_FOR_DRAWBACK")
        exit_row = signal_lookup(signal_summary, "EXIT")

        if breakout is not None or pullback is not None or prepare is not None:
            handle.write("## Readout\n\n")
            if breakout is not None:
                handle.write(
                    f"- Breakout adds produced a median 60-day return of {breakout['median_return_60d']:.1%} with a median 60-day drawdown of {breakout['median_max_drawdown_60d']:.1%}.\n"
                )
            if pullback is not None:
                handle.write(
                    f"- Pullback adds produced a median 60-day return of {pullback['median_return_60d']:.1%} with a median 60-day drawdown of {pullback['median_max_drawdown_60d']:.1%}.\n"
                )
            if prepare is not None:
                handle.write(
                    f"- Prepare signals did not mean immediate exit. They mostly flagged hotter, riskier stretches that still saw a median 60-day drawdown of {prepare['median_max_drawdown_60d']:.1%}.\n"
                )
            if exit_row is not None:
                handle.write(
                    f"- Exit is intentionally a late capital-preservation trigger, not a top-tick rule. In this first pass it still allowed a median 60-day drawdown of {exit_row['median_max_drawdown_60d']:.1%}, so it should be treated as the weakest component and improved next.\n"
                )
            handle.write("\n")

        handle.write("## Top Clean Bull Runs\n\n")
        handle.write("```\n")
        handle.write(format_table(top_clean, max_rows=15))
        handle.write("\n```\n\n")

        if not events.empty:
            signal_examples = (
                events.sort_values(["signal", "max_gain_60d"], ascending=[True, False])
                .groupby("signal", as_index=False)
                .head(5)[
                    [
                        "symbol",
                        "signal",
                        "date",
                        "close",
                        "return_20d",
                        "return_60d",
                        "max_drawdown_20d",
                        "max_drawdown_60d",
                    ]
                ]
            )
            handle.write("## Signal Examples\n\n")
            handle.write("```\n")
            handle.write(format_table(signal_examples, max_rows=20))
            handle.write("\n```\n\n")

        if not latest_actions.empty:
            handle.write("## Latest System Snapshot\n\n")
            handle.write("```\n")
            handle.write(
                format_table(
                    latest_actions.sort_values(
                        ["latest_action", "max_return_factor"],
                        ascending=[True, False],
                    ),
                    max_rows=20,
                )
            )
            handle.write("\n```\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.input_csv, args.limit)
    symbols = candidates["symbol"].tolist()
    histories, failed_downloads = download_histories(
        symbols=symbols,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
    )

    candidate_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    latest_snapshots: list[pd.DataFrame] = []

    for symbol in symbols:
        history = histories.get(symbol)
        if history is None or history.empty:
            continue

        assessed = assess_candidate(symbol, history, args)
        if assessed is None:
            continue

        row = {
            **assessed,
            "source_market_cap": float(
                candidates.loc[candidates["symbol"] == symbol, "marketCap"].iloc[0]
            )
            if "marketCap" in candidates.columns
            else np.nan,
            "source_industry": (
                candidates.loc[candidates["symbol"] == symbol, "industry"].iloc[0]
                if "industry" in candidates.columns
                else ""
            ),
        }
        candidate_rows.append(row)

        if not row["quality_pass"]:
            continue

        clean_rows.append(row)
        events, latest = build_signal_table(symbol, history, row, args)
        if not events.empty:
            event_frames.append(events)
        latest_snapshots.append(latest)

    assessed_candidates = pd.DataFrame(candidate_rows).sort_values(
        ["quality_pass", "max_return_factor"], ascending=[False, False]
    )
    clean_candidates = pd.DataFrame(clean_rows).sort_values(
        ["max_return_factor", "median_dollar_volume_run"], ascending=[False, False]
    )
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    latest_actions = (
        pd.concat(latest_snapshots, ignore_index=True) if latest_snapshots else pd.DataFrame()
    )
    signal_summary = summarize_signals(events)

    assessed_candidates.to_csv(
        args.output_dir / "bull_run_candidates.csv", index=False
    )
    clean_candidates.to_csv(args.output_dir / "bull_run_candidates_clean.csv", index=False)
    events.to_csv(args.output_dir / "bull_run_signal_events.csv", index=False)
    signal_summary.to_csv(args.output_dir / "bull_run_signal_summary.csv", index=False)
    latest_actions.to_csv(args.output_dir / "bull_run_latest_actions.csv", index=False)

    write_report(
        output_dir=args.output_dir,
        candidates=candidates,
        clean_candidates=clean_candidates,
        failed_downloads=failed_downloads,
        signal_summary=signal_summary,
        events=events,
        latest_actions=latest_actions,
    )

    print(f"Analyzed {len(candidates)} input candidates.")
    print(f"Clean 10-baggers: {len(clean_candidates)}")
    print(f"Signal events: {len(events)}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
