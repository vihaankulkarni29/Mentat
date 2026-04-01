"""Beginner-friendly reporting for Mentat."""

from __future__ import annotations

import json
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src import config

REGIME_EMOJI = {
    "LOW-VOL TRENDING": "UPTREND",
    "MEAN-REVERTING": "RANGE",
    "HIGH-VOL RANGING": "HIGH VOL",
    "CRASH/CRISIS": "RISK",
    "UNCERTAIN / TRANSITION": "UNCERTAIN",
}


def build_report(results: dict[str, dict]) -> str:
    """Build a plain-language daily report for beginners."""
    today = date.today().strftime("%d %b %Y")
    lines = [f"=== MENTAT DAILY REPORT - {today} ===", ""]

    for ticker, data in results.items():
        regime = data["regime_label"]
        base_regime = data.get("base_regime_label", regime)
        regime_confidence = data.get("regime_confidence", 0.0)
        probs = data["state_probs"]
        risk = data["risk"]
        outliers = data["outliers"]
        history = data.get("regime_history", [])
        persistence = data.get("persistence", {})
        model_quality = data.get("model_quality")
        regime_tag = REGIME_EMOJI.get(regime, "NEUTRAL")

        lines.append("-" * 56)
        lines.append(f"{ticker}")
        lines.append(f"Regime: {regime} ({regime_tag})")
        lines.append(f"Regime confidence: {regime_confidence:.0%} | Base regime: {base_regime}")
        lines.append("State confidence: " + " | ".join(f"S{i}: {p:.0%}" for i, p in enumerate(probs)))
        lines.append(
            f"Risk snapshot -> VaR95: {risk['regime_var_95']:.2%}, "
            f"CVaR95: {risk['regime_cvar_95']:.2%}, Sharpe: {risk['regime_sharpe']}, Beta: {risk['beta']}"
        )
        lines.append("Risk explained: " + _explain_var(risk["regime_var_95"]))
        lines.append("               " + _explain_sharpe(float(risk["regime_sharpe"])))
        lines.append("               " + _explain_beta(float(risk["beta"])))

        if persistence:
            lines.append(
                "Regime persistence: "
                f"{persistence.get('days_in_regime', 0)} day(s) in current state, "
                f"{persistence.get('regime_maturity', 'UNKNOWN')}"
            )

        if model_quality:
            lines.append(
                "Model quality: "
                f"LL={model_quality.get('log_likelihood')}, "
                f"states_used={model_quality.get('states_used')}, "
                f"min_persistence={model_quality.get('min_persistence')}"
            )

        if outliers:
            lines.append("Alerts: " + json.dumps(outliers))
        else:
            lines.append("Alerts: none")

        lines.append("Action hint: " + _action_hint_from_regime(regime))
        if history:
            lines.append("Last 5 sessions:")
            for row in history:
                lines.append(
                    f"  - {row['date']}: {row['regime']} "
                    f"(state={row['state']}, conf={row['confidence']:.0%})"
                )
        lines.append("")

    lines.append("-" * 56)
    lines.append("Informational output only; final decisions remain yours.")
    return "\n".join(lines)


def _action_hint_from_regime(regime: str) -> str:
    if regime == "LOW-VOL TRENDING":
        return "Momentum-friendly. Favor trend-following entries with strict risk limits."
    if regime == "HIGH-VOL RANGING":
        return "Choppy regime. Reduce position size and avoid overtrading."
    if regime == "CRASH/CRISIS":
        return "Defense first. Hedge or de-risk until regime stabilizes."
    if regime == "MEAN-REVERTING":
        return "Counter-trend setups can work, but require disciplined stops."
    return "No clear edge detected today."


def _explain_var(var_val: float) -> str:
    pct = abs(var_val) * 100
    return f"On a bad day (5% tail), this stock could fall about {pct:.1f}%."


def _explain_beta(beta: float) -> str:
    if beta < 0.8:
        return f"Beta {beta:.2f}: moves less than market; relatively defensive."
    if beta < 1.2:
        return f"Beta {beta:.2f}: moves roughly in line with market."
    return f"Beta {beta:.2f}: amplifies market moves; higher directional risk."


def _explain_sharpe(sharpe: float) -> str:
    if sharpe < 0:
        return f"Sharpe {sharpe:.2f}: return has not compensated risk."
    if sharpe < 0.5:
        return f"Sharpe {sharpe:.2f}: weak risk-adjusted return."
    if sharpe < 1.0:
        return f"Sharpe {sharpe:.2f}: acceptable risk-adjusted return."
    return f"Sharpe {sharpe:.2f}: strong risk-adjusted return."


def save_report(report_text: str) -> str:
    """Persist report to disk."""
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    filename = f"mentat_report_{date.today().isoformat()}.txt"
    path = os.path.join(config.REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path


def send_email(report_text: str) -> None:
    """Optional email dispatch; guarded by config and env password."""
    if not config.SEND_EMAIL or not config.REPORT_EMAIL:
        return

    password = os.getenv(config.SMTP_PASSWORD_ENV, "")
    if not password:
        raise RuntimeError(f"Missing SMTP password env var: {config.SMTP_PASSWORD_ENV}")

    msg = MIMEMultipart()
    msg["From"] = config.REPORT_EMAIL
    msg["To"] = config.REPORT_EMAIL
    msg["Subject"] = f"Mentat Report - {date.today().isoformat()}"
    msg.attach(MIMEText(report_text, "plain"))

    with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.login(config.REPORT_EMAIL, password)
        server.sendmail(config.REPORT_EMAIL, config.REPORT_EMAIL, msg.as_string())
