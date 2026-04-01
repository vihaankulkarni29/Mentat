"""Mentat Streamlit dashboard (Phase 1.1)."""

from __future__ import annotations

import random
from typing import Dict

import pandas as pd
import streamlit as st

import pipeline
from src import config

DUNE_CSS = """
<style>
:root {
  --bg: #1f120a;
  --panel: #2b190d;
  --panel-soft: #3a2413;
  --sand: #d2b48c;
  --spice: #e07a2f;
  --mint: #a9d6b6;
  --alert: #d95d39;
}

.stApp {
  background: linear-gradient(180deg, #1b1008 0%, #241408 40%, #2d1808 100%);
  color: var(--sand);
}

.block-container {
  padding-top: 1.2rem;
}

.mentat-card {
  background: var(--panel);
  border: 1px solid #5a351b;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 0.8rem;
}

.mentat-title {
  color: var(--spice);
  font-weight: 700;
  font-size: 1.1rem;
}

.mentat-sub {
  color: var(--sand);
  opacity: 0.9;
  font-size: 0.95rem;
}

.mentat-tag {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.8rem;
  margin-right: 0.4rem;
  background: #4a2b16;
  color: #f2d7b5;
}

.mentat-quote {
  background: var(--panel-soft);
  border-left: 4px solid var(--spice);
  padding: 0.8rem;
  border-radius: 8px;
  margin: 0.5rem 0 1rem 0;
  font-style: italic;
}
</style>
"""

MENTAT_SAYINGS = {
    "LOW-VOL TRENDING": [
        "The pattern is coherent. Ride the flow, but never abandon discipline.",
        "Momentum has memory. Respect it, and size with restraint.",
    ],
    "HIGH-VOL RANGING": [
        "Noise is loud today. Precision matters more than speed.",
        "When volatility expands, reduce ego and reduce size.",
    ],
    "CRASH/CRISIS": [
        "Preservation is profit in slow motion.",
        "In disorder, your first edge is survival.",
    ],
    "MEAN-REVERTING": [
        "Extremes often bend back toward balance.",
        "Counter-trend can work, but only with disciplined exits.",
    ],
    "UNCERTAIN / TRANSITION": [
        "Signals conflict. Patience is a strategic position.",
        "When certainty is low, exposure should be low.",
    ],
}


def _say(regime: str) -> str:
    choices = MENTAT_SAYINGS.get(regime, ["Observe first, act second."])
    return random.choice(choices)


def _regime_color(regime: str) -> str:
    if regime == "LOW-VOL TRENDING":
        return "#6bbf8f"
    if regime == "MEAN-REVERTING":
        return "#d2b48c"
    if regime == "HIGH-VOL RANGING":
        return "#e09f3e"
    if regime == "CRASH/CRISIS":
        return "#d95d39"
    return "#c08c5a"


def render_header() -> None:
    st.markdown(DUNE_CSS, unsafe_allow_html=True)
    st.title("Mentat: Regime Intelligence Console")
    st.caption("Hidden Markov decision support for a changing market")
    st.markdown(
        '<div class="mentat-quote">"A Mentat computes possibility, but you decide destiny."</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[list[str], bool]:
    st.sidebar.header("Mission Controls")
    selected = st.sidebar.multiselect("Tickers", options=config.TICKERS, default=config.TICKERS)
    retrain = st.sidebar.checkbox("Retrain HMM models", value=False)
    st.sidebar.caption("Tip: Retrain weekly, infer daily.")
    return selected, retrain


def render_ticker_card(ticker: str, payload: Dict) -> None:
    regime = payload["regime_label"]
    base_regime = payload.get("base_regime_label", regime)
    conf = payload.get("regime_confidence", 0.0)
    risk = payload["risk"]
    outliers = payload.get("outliers", {})
    history = payload.get("regime_history", [])

    tag_color = _regime_color(regime)

    st.markdown('<div class="mentat-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="mentat-title">{ticker}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="mentat-tag" style="background:{tag_color};color:#1f120a;">{regime}</span>'
        f'<span class="mentat-tag">Confidence: {conf:.0%}</span>'
        f'<span class="mentat-tag">Base: {base_regime}</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95%", f"{risk['regime_var_95']:.2%}")
    c2.metric("CVaR 95%", f"{risk['regime_cvar_95']:.2%}")
    c3.metric("Sharpe", f"{risk['regime_sharpe']}")
    c4.metric("Beta", f"{risk['beta']}")

    st.markdown(
        f'<div class="mentat-quote">Mentat says: {_say(regime)}</div>',
        unsafe_allow_html=True,
    )

    if outliers:
        st.warning(f"Outlier alerts: {outliers}")
    else:
        st.success("No outlier alerts for today.")

    if history:
        st.markdown('<div class="mentat-sub">Last 5 sessions regime tape</div>', unsafe_allow_html=True)
        history_df = pd.DataFrame(history)
        history_df = history_df[["date", "regime", "state", "confidence"]]
        history_df.columns = ["Date", "Regime", "State", "Confidence"]
        st.dataframe(history_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    render_header()
    selected_tickers, retrain = render_sidebar()

    if not selected_tickers:
        st.info("Choose at least one ticker from the sidebar.")
        return

    if st.button("Run Mentat Analysis", type="primary"):
        with st.spinner("Mentat is computing regimes, risk, and transitions..."):
            results = pipeline.run_pipeline(retrain=retrain, tickers=selected_tickers)

        st.subheader("Daily Decision Surface")
        for ticker in selected_tickers:
            if ticker in results:
                render_ticker_card(ticker, results[ticker])
            else:
                st.error(f"No data available for {ticker} in this run.")


if __name__ == "__main__":
    main()
