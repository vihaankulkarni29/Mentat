"""Mentat morning brief — the Emperor's daily intelligence report."""

from __future__ import annotations

from datetime import date

MENTAT_VOICE = {
    "BULLISH_HIGH": "My calculations show strong accumulation. The data supports entry.",
    "BULLISH_MED": "Signals are constructive. Position sizing should remain measured.",
    "BEARISH": "Adverse conditions detected. Preservation takes priority.",
    "NEUTRAL": "No clear edge today. Patience is the correct position.",
    "UNCERTAIN": "Conflicting signals. I advise waiting for clarity.",
}


def build_morning_brief(
    pipeline_results: dict,
    sentiment_data: dict,
    market_regime: dict | None = None,
) -> str:
    """Build the daily morning brief combining regime, outliers, and sentiment."""
    today = date.today().strftime("%A, %d %B %Y")
    lines = []

    # ── Header ──
    lines += [
        "=" * 50,
        "MENTAT MORNING BRIEF",
        f"{today}",
        "=" * 50,
        "",
        "Good morning, Master Vihaan.",
        "",
    ]

    # ── Market mood ──
    n_crisis = sum(
        1
        for r in pipeline_results.values()
        if r.get("regime_label") == "CRASH/CRISIS"
    )
    n_trending = sum(
        1
        for r in pipeline_results.values()
        if r.get("regime_label") == "LOW-VOL TRENDING"
    )
    n_total = len(pipeline_results)

    if n_total > 0:
        crisis_pct = n_crisis / n_total
    else:
        crisis_pct = 0

    if crisis_pct > 0.6:
        mood = "DEFENSIVE — majority of your watchlist is in crisis regime."
        mood_hint = MENTAT_VOICE["BEARISH"]
    elif n_trending > 0 and n_trending / max(n_total, 1) > 0.5:
        mood = "CONSTRUCTIVE — trending conditions dominate."
        mood_hint = MENTAT_VOICE["BULLISH_MED"]
    else:
        mood = "MIXED — market is regime-fragmented."
        mood_hint = MENTAT_VOICE["NEUTRAL"]

    lines += [
        "-- MARKET INTELLIGENCE -" + "-" * 20,
        f"Overall mood: {mood}",
        f"Mentat assessment: {mood_hint}",
        f"Stocks scanned: {n_total} | In crisis: {n_crisis} | Trending: {n_trending}",
        "",
    ]

    # ── Top outliers ──
    # Score each stock: regime score + outlier score + sentiment boost
    scored = []
    for ticker, data in pipeline_results.items():
        regime = data.get("regime_label", "UNCERTAIN")
        conf = data.get("regime_confidence", 0.5)
        persist = data.get("persistence", {}).get("days_in_regime", 0)
        outliers = data.get("outliers", {})
        risk = data.get("risk", {})

        # Base score from regime
        regime_score = {
            "LOW-VOL TRENDING": 3,
            "MEAN-REVERTING": 1,
            "HIGH-VOL RANGING": 0,
            "CRASH/CRISIS": -2,
            "UNCERTAIN": -1,
        }.get(regime, 0)

        # Outlier signal boost
        outlier_boost = min(len(outliers), 2)

        # Sentiment boost from news
        sentiment_signals = sentiment_data.get("by_ticker", {}).get(ticker, [])
        sentiment_boost = sum(
            s["confidence"] / 10.0
            for s in sentiment_signals
            if s.get("direction") == "BULLISH"
        )
        sentiment_drag = sum(
            s["confidence"] / 10.0
            for s in sentiment_signals
            if s.get("direction") == "BEARISH"
        )

        conviction = (
            regime_score * conf + outlier_boost * 0.5 + sentiment_boost - sentiment_drag
        )

        scored.append(
            {
                "ticker": ticker,
                "conviction": round(conviction, 2),
                "regime": regime,
                "conf": conf,
                "persist": persist,
                "sharpe": risk.get("regime_sharpe", 0),
                "var": risk.get("regime_var_95", 0),
                "outliers": outliers,
                "sentiment": sentiment_signals,
                "risk": risk,
            }
        )

    # Sort: highest conviction first, exclude crash regime
    top5 = sorted(
        [s for s in scored if s["regime"] != "CRASH/CRISIS"],
        key=lambda x: x["conviction"],
        reverse=True,
    )[:5]

    watchlist = sorted(
        [s for s in scored if s["regime"] == "CRASH/CRISIS"],
        key=lambda x: x["conviction"],
        reverse=True,
    )

    lines.append("-- TOP OUTLIERS TODAY -" + "-" * 21)
    if not top5:
        lines.append("No high-conviction setups today. Mentat advises patience.")
    else:
        for i, stock in enumerate(top5, 1):
            lines += [
                "",
                f"  {i}. {stock['ticker']}",
                f"     Regime: {stock['regime']} | Confidence: {stock['conf']:.0%}"
                f" | In regime: {stock['persist']} days",
                f"     Sharpe: {stock['sharpe']} | VaR 95%: {stock['var']:.2%}",
                f"     Conviction score: {stock['conviction']:.1f}/10",
            ]
            if stock["outliers"]:
                flag_keys = [
                    k for k in stock["outliers"] if not k.endswith("_recent")
                ][:2]
                lines.append(f"     Outlier flags: {', '.join(flag_keys)}")
            if stock["sentiment"]:
                for sig in stock["sentiment"][:2]:
                    lines.append(
                        f"     News [{sig['direction']} {sig['confidence']}/10]: "
                        f"{sig['reason']}"
                    )

    # ── Crisis watch ──
    if watchlist:
        lines += [
            "",
            "-- CRISIS / AVOID TODAY -" + "-" * 19,
        ]
        for stock in watchlist[:3]:
            lines.append(
                f"  {stock['ticker']}: {stock['regime']} "
                f"| {stock['persist']}d | VaR {stock['var']:.2%}"
            )

    # ── Sentiment summary ──
    n_signals = sentiment_data.get("n_articles", 0)
    all_sigs = sentiment_data.get("signals", [])
    bull_sigs = [s for s in all_sigs if s.get("direction") == "BULLISH"]
    bear_sigs = [s for s in all_sigs if s.get("direction") == "BEARISH"]

    lines += [
        "",
        "-- NEWS INTELLIGENCE -" + "-" * 23,
        f"  Articles processed: {n_signals}",
        f"  Bullish catalysts:  {len(bull_sigs)}",
        f"  Bearish catalysts:  {len(bear_sigs)}",
    ]
    for sig in (bull_sigs + bear_sigs)[:4]:
        lines.append(
            f"  [{sig['direction']}] {sig.get('ticker', '?')}: "
            f"{sig.get('reason', '')}"
        )

    # ── Closing ──
    lines += [
        "",
        "=" * 50,
        "The calculations are complete. The decision is yours.",
        f"Mentat signs off -- {today}",
        "=" * 50,
    ]

    return "\n".join(lines)
