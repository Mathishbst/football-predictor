"""
⚽ Backtesting — Simulation de paris sur les matchs de test
============================================================
Stratégies disponibles :
  1. Value Betting   : parie si la proba du modèle > proba implicite bookmaker
  2. Confiance élevée: parie uniquement quand le modèle est très sûr (> seuil)
  3. Kelly Criterion : mise proportionnelle à l'avantage perçu
"""

import numpy as np
import pandas as pd


# ===========================================================================
# UTILITAIRES
# ===========================================================================

def implied_prob(odd: float) -> float:
    """Cote décimale → probabilité implicite bookmaker."""
    return 1 / odd if odd > 0 else np.nan


def kelly_fraction(prob_model: float, odd: float, fraction: float = 0.25) -> float:
    """
    Fraction de Kelly (fractionnée pour limiter le risque).
    f = fraction * (p * odd - 1) / (odd - 1)
    Retourne 0 si l'espérance est négative.
    """
    edge = prob_model * odd - 1
    if edge <= 0 or odd <= 1:
        return 0.0
    f = fraction * edge / (odd - 1)
    return min(f, 0.25)   # cap à 25% de la bankroll max


# ===========================================================================
# BACKTESTING PRINCIPAL
# ===========================================================================

def run_backtest(
    trained: dict,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,              # DataFrame original (pour les cotes)
    strategy: str = "value",           # "value" | "confidence" | "kelly"
    confidence_threshold: float = 0.55,
    stake_pct: float = 0.02,           # mise fixe = 2% de la bankroll
    initial_bankroll: float = 1000.0,
) -> dict:
    """
    Simule une saison de paris et retourne les résultats détaillés.

    Paramètres
    ----------
    strategy :
        "value"      → parie si edge > 0 (proba modèle > proba bookmaker)
        "confidence" → parie si max(proba) > confidence_threshold
        "kelly"      → mise variable selon critère de Kelly fractionné
    """
    b = trained[model_name]

    # ── Probabilités du modèle ────────────────────────────────────────────
    X = b["scaler"].transform(X_test) if b["scaler"] else X_test
    probas = b["model"].predict_proba(X)           # shape (n, 3)

    if b["le"] is not None:
        classes = list(b["le"].classes_)
    else:
        classes = list(b["model"].classes_)

    idx_H = classes.index("H")
    idx_D = classes.index("D")
    idx_A = classes.index("A")

    # ── Cotes bookmaker ───────────────────────────────────────────────────
    odds_cols = {"H": "B365H", "D": "B365D", "A": "B365A"}
    has_odds  = all(c in df_test.columns for c in odds_cols.values())

    results  = []
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = 0
    bets_won    = 0

    for i in range(len(X_test)):
        p_H = probas[i, idx_H]
        p_D = probas[i, idx_D]
        p_A = probas[i, idx_A]
        actual = y_test.iloc[i]

        row = df_test.iloc[i]
        odd_H = row.get("B365H", np.nan) if has_odds else np.nan
        odd_D = row.get("B365D", np.nan) if has_odds else np.nan
        odd_A = row.get("B365A", np.nan) if has_odds else np.nan

        # ── Sélection du pari selon stratégie ────────────────────────────
        bet_outcome = None
        bet_odd     = np.nan
        bet_prob    = 0.0
        stake       = 0.0

        candidates = [
            ("H", p_H, odd_H),
            ("D", p_D, odd_D),
            ("A", p_A, odd_A),
        ]

        if strategy == "value" and has_odds:
            # Parie sur l'outcome avec le meilleur edge positif
            best_edge, best = -999, None
            for outcome, prob, odd in candidates:
                if pd.isna(odd) or odd <= 1:
                    continue
                edge = prob - implied_prob(odd)
                if edge > best_edge:
                    best_edge, best = edge, (outcome, prob, odd)
            if best and best_edge > 0:
                bet_outcome, bet_prob, bet_odd = best
                stake = bankroll * stake_pct

        elif strategy == "confidence":
            # Parie sur l'outcome le plus probable si confiance suffisante
            best = max(candidates, key=lambda x: x[1])
            if best[1] >= confidence_threshold:
                bet_outcome, bet_prob, bet_odd = best
                stake = bankroll * stake_pct

        elif strategy == "kelly" and has_odds:
            # Mise variable par Kelly sur le meilleur value bet
            best_edge, best = -999, None
            for outcome, prob, odd in candidates:
                if pd.isna(odd) or odd <= 1:
                    continue
                edge = prob - implied_prob(odd)
                if edge > best_edge:
                    best_edge, best = edge, (outcome, prob, odd)
            if best and best_edge > 0:
                bet_outcome, bet_prob, bet_odd = best
                f = kelly_fraction(bet_prob, bet_odd)
                stake = bankroll * f

        # ── Calcul du P&L ─────────────────────────────────────────────────
        pnl = 0.0
        if bet_outcome and stake > 0 and bankroll > 0:
            bets_placed += 1
            stake = min(stake, bankroll)
            if actual == bet_outcome:
                pnl = stake * (bet_odd - 1)
                bets_won += 1
            else:
                pnl = -stake
            bankroll += pnl

        bankroll_history.append(bankroll)
        results.append({
            "match":       f"{row.get('HomeTeam','?')} vs {row.get('AwayTeam','?')}",
            "actual":      actual,
            "pred_H":      round(p_H, 3),
            "pred_D":      round(p_D, 3),
            "pred_A":      round(p_A, 3),
            "bet":         bet_outcome or "—",
            "odd":         round(bet_odd, 2) if not pd.isna(bet_odd) else "—",
            "stake":       round(stake, 2),
            "pnl":         round(pnl, 2),
            "bankroll":    round(bankroll, 2),
        })

    df_results = pd.DataFrame(results)
    total_pnl  = bankroll - initial_bankroll
    roi        = total_pnl / (df_results["stake"].sum() or 1) * 100

    summary = {
        "model":            model_name,
        "strategy":         strategy,
        "initial_bankroll": initial_bankroll,
        "final_bankroll":   round(bankroll, 2),
        "total_pnl":        round(total_pnl, 2),
        "roi_pct":          round(roi, 2),
        "bets_placed":      bets_placed,
        "bets_won":         bets_won,
        "win_rate":         round(bets_won / bets_placed * 100, 1) if bets_placed else 0,
        "max_drawdown":     round(_max_drawdown(bankroll_history), 2),
        "bankroll_history": bankroll_history,
        "details":          df_results,
    }
    return summary


def _max_drawdown(history: list) -> float:
    """Calcule le drawdown maximum (perte max depuis un pic)."""
    peak = history[0]
    max_dd = 0.0
    for v in history:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def print_summary(s: dict):
    sign = "+" if s["total_pnl"] >= 0 else ""
    print(f"\n{'='*50}")
    print(f"📊 Backtesting — {s['model']} | Stratégie : {s['strategy']}")
    print(f"{'='*50}")
    print(f"  Matchs analysés   : {len(s['details'])}")
    print(f"  Paris placés      : {s['bets_placed']}")
    print(f"  Paris gagnés      : {s['bets_won']}  ({s['win_rate']}%)")
    print(f"  Bankroll initiale : {s['initial_bankroll']:.0f} €")
    print(f"  Bankroll finale   : {s['final_bankroll']:.2f} €")
    print(f"  P&L total         : {sign}{s['total_pnl']:.2f} €")
    print(f"  ROI               : {sign}{s['roi_pct']:.2f}%")
    print(f"  Drawdown max      : -{s['max_drawdown']:.2f} €")
