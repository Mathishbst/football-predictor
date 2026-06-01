"""
⚽ Football Predictor — Interface Streamlit
==========================================
Lancement :
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pipeline import load_all_seasons, clean, build_features, train, predict
from backtesting import run_backtest, print_summary

# ===========================================================================
# CONFIG
# ===========================================================================

DATA_DIR = "data"
MODEL_DIR = "models"

st.set_page_config(
    page_title="⚽ Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #1e4d8c;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
    .prob-bar-H { background: #2ecc71; }
    .prob-bar-D { background: #f39c12; }
    .prob-bar-A { background: #e74c3c; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# CHARGEMENT & CACHE
# ===========================================================================

@st.cache_data(show_spinner="📦 Chargement des données...")
def load_data():
    raw = load_all_seasons(DATA_DIR)
    return clean(raw)


@st.cache_resource(show_spinner="🧠 Entraînement des modèles...")
def load_models(data_hash: int):
    df = load_data()
    X, y = build_features(df)
    split = int(len(X) * 0.80)
    trained = train(X.iloc[:split], y.iloc[:split])
    return trained, X, y, df


@st.cache_data(show_spinner="🔄 Backtesting en cours...")
def compute_backtest(model_name, strategy, threshold, stake_pct, data_hash):
    df = load_data()
    X, y = build_features(df)
    split = int(len(X) * 0.80)
    trained, _, _, _ = load_models(data_hash)
    X_test  = X.iloc[split:]
    y_test  = y.iloc[split:]
    df_test = df.iloc[split:].reset_index(drop=True)
    return run_backtest(
        trained, model_name, X_test, y_test, df_test,
        strategy=strategy,
        confidence_threshold=threshold,
        stake_pct=stake_pct,
    )


# ===========================================================================
# HELPERS VISUELS
# ===========================================================================

def prob_gauge(label: str, value: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": label, "font": {"size": 14, "color": "#ccc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#333",
            "steps": [
                {"range": [0,  33], "color": "#1a1a2e"},
                {"range": [33, 66], "color": "#16213e"},
                {"range": [66, 100],"color": "#0f3460"},
            ],
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def bankroll_chart(history: list, initial: float):
    x = list(range(len(history)))
    colors = ["#2ecc71" if v >= initial else "#e74c3c" for v in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=history,
        mode="lines",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.08)",
        name="Bankroll",
    ))
    fig.add_hline(y=initial, line_dash="dash", line_color="#888",
                  annotation_text="Bankroll initiale")
    fig.update_layout(
        title="Évolution de la bankroll",
        xaxis_title="Matchs",
        yaxis_title="Bankroll (€)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222"),
        height=350,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def results_donut(bets_won, bets_placed):
    bets_lost = bets_placed - bets_won
    fig = go.Figure(go.Pie(
        labels=["Gagnés", "Perdus"],
        values=[bets_won, bets_lost],
        hole=0.6,
        marker_colors=["#2ecc71", "#e74c3c"],
        textinfo="label+percent",
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def pnl_bar_chart(df_details: pd.DataFrame):
    df = df_details[df_details["stake"] > 0].copy()
    df["color"] = df["pnl"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")
    fig = go.Figure(go.Bar(
        x=list(range(len(df))),
        y=df["pnl"],
        marker_color=df["color"],
        name="P&L par pari",
    ))
    fig.update_layout(
        title="P&L par pari",
        xaxis_title="Pari #",
        yaxis_title="Gain / Perte (€)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222"),
        height=300,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


# ===========================================================================
# APP PRINCIPALE
# ===========================================================================

def main():
    # ── Chargement ────────────────────────────────────────────────────────
    df = load_data()
    data_hash = len(df)
    trained, X, y, df_full = load_models(data_hash)

    split     = int(len(X) * 0.80)
    X_test    = X.iloc[split:]
    y_test    = y.iloc[split:]
    df_test   = df_full.iloc[split:].reset_index(drop=True)

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/1200px-Premier_League_Logo.svg.png",
                 width=80)
        st.title("⚽ Football Predictor")
        st.caption(f"{len(df)} matchs · {df['Date'].min().year}–{df['Date'].max().year}")
        st.divider()
        model_name = st.selectbox("🧠 Modèle", list(trained.keys()), index=0)
        st.divider()
        st.caption("Made with ❤️ + ML")

    # ── Onglets ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📈 Backtesting", "📊 Données"])

    # ======================================================================
    # TAB 1 — PRÉDICTION
    # ======================================================================
    with tab1:
        st.header("🔮 Prédire un match")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🏠 Équipe domicile")
            home_team = st.selectbox("", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0, key="home")
        with col2:
            st.markdown("### ✈️ Équipe extérieure")
            away_options = [t for t in teams if t != home_team]
            away_team = st.selectbox("", away_options,
                                     index=away_options.index("Man City") if "Man City" in away_options else 0,
                                     key="away")

        st.divider()

        # Stats des équipes basées sur les derniers matchs réels
        def get_team_recent_stats(team, df, n=10):
            home_m = df[df["HomeTeam"] == team].tail(n)
            away_m = df[df["AwayTeam"] == team].tail(n)

            pts, scored, conceded, shots, shots_on = [], [], [], [], []
            wins = []

            for _, r in home_m.iterrows():
                pts.append(3 if r["FTR"]=="H" else (1 if r["FTR"]=="D" else 0))
                scored.append(r["FTHG"]); conceded.append(r["FTAG"])
                shots.append(r.get("HS", 11)); shots_on.append(r.get("HST", 4))
                wins.append(int(r["FTR"]=="H"))
            for _, r in away_m.iterrows():
                pts.append(3 if r["FTR"]=="A" else (1 if r["FTR"]=="D" else 0))
                scored.append(r["FTAG"]); conceded.append(r["FTHG"])
                shots.append(r.get("AS", 11)); shots_on.append(r.get("AST", 4))
                wins.append(int(r["FTR"]=="A"))

            if not pts:
                return dict(form=0.45, scored=1.4, conceded=1.4,
                            win_rate=0.35, shots=11.0, shots_on=4.0)
            return dict(
                form=np.mean(pts)/3, scored=np.mean(scored),
                conceded=np.mean(conceded), win_rate=np.mean(wins),
                shots=np.mean(shots), shots_on=np.mean(shots_on),
            )

        hs = get_team_recent_stats(home_team, df)
        as_ = get_team_recent_stats(away_team, df)

        # Affichage des stats des équipes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 📋 Stats récentes — {home_team}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Forme", f"{hs['form']:.0%}")
            c2.metric("Buts/match", f"{hs['scored']:.1f}")
            c3.metric("Taux victoires", f"{hs['win_rate']:.0%}")
        with col2:
            st.markdown(f"#### 📋 Stats récentes — {away_team}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Forme", f"{as_['form']:.0%}")
            c2.metric("Buts/match", f"{as_['scored']:.1f}")
            c3.metric("Taux victoires", f"{as_['win_rate']:.0%}")

        st.divider()

        if st.button("🔮 Lancer la prédiction", type="primary", use_container_width=True):
            features = {
                "home_form":      hs["form"],
                "away_form":      as_["form"],
                "home_scored":    hs["scored"],
                "home_conceded":  hs["conceded"],
                "away_scored":    as_["scored"],
                "away_conceded":  as_["conceded"],
                "home_win_rate":  hs["win_rate"],
                "away_win_rate":  as_["win_rate"],
                "home_shots":     hs["shots"],
                "away_shots":     as_["shots"],
                "home_shots_on":  hs["shots_on"],
                "away_shots_on":  as_["shots_on"],
                "form_diff":      hs["form"] - as_["form"],
                "scored_diff":    hs["scored"] - as_["scored"],
                "conceded_diff":  hs["conceded"] - as_["conceded"],
                "shots_diff":     hs["shots"] - as_["shots"],
                "win_rate_diff":  hs["win_rate"] - as_["win_rate"],
            }

            b = trained[model_name]
            X_pred = pd.DataFrame([{c: features.get(c, 0) for c in X.columns}]).fillna(0)
            X_input = b["scaler"].transform(X_pred) if b["scaler"] else X_pred
            probas = b["model"].predict_proba(X_input)[0]

            if b["le"] is not None:
                classes = list(b["le"].classes_)
            else:
                classes = list(b["model"].classes_)

            p = dict(zip(classes, probas))
            p_H, p_D, p_A = p.get("H", 0), p.get("D", 0), p.get("A", 0)

            st.markdown(f"## {home_team} 🆚 {away_team}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(prob_gauge("🏠 Victoire domicile", p_H, "#2ecc71"),
                                use_container_width=True)
            with col2:
                st.plotly_chart(prob_gauge("🤝 Nul", p_D, "#f39c12"),
                                use_container_width=True)
            with col3:
                st.plotly_chart(prob_gauge("✈️ Victoire extérieure", p_A, "#e74c3c"),
                                use_container_width=True)

            winner = max({"H": p_H, "D": p_D, "A": p_A}, key=lambda k: {"H": p_H, "D": p_D, "A": p_A}[k])
            labels = {"H": f"🏠 Victoire {home_team}", "D": "🤝 Match nul", "A": f"✈️ Victoire {away_team}"}
            proba_winner = {"H": p_H, "D": p_D, "A": p_A}[winner]

            st.success(f"**Pronostic : {labels[winner]}** — confiance {proba_winner:.1%}")

    # ======================================================================
    # TAB 2 — BACKTESTING
    # ======================================================================
    with tab2:
        st.header("📈 Backtesting")
        st.caption(f"Simulation sur {len(X_test)} matchs de test (20% les plus récents)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strategy = st.selectbox("Stratégie", ["value", "confidence", "kelly"],
                                    format_func=lambda x: {
                                        "value": "💡 Value Betting",
                                        "confidence": "🎯 Haute confiance",
                                        "kelly": "📐 Kelly Criterion"
                                    }[x])
        with col2:
            threshold = st.slider("Seuil confiance", 0.40, 0.80, 0.55, 0.01,
                                  disabled=(strategy != "confidence"))
        with col3:
            stake_pct = st.slider("Mise fixe (%)", 1, 10, 2,
                                  disabled=(strategy == "kelly")) / 100
        with col4:
            initial_br = st.number_input("Bankroll initiale (€)", 100, 10000, 1000, 100)

        if st.button("🚀 Lancer le backtesting", type="primary", use_container_width=True):
            with st.spinner("Simulation en cours..."):
                result = run_backtest(
                    trained, model_name, X_test, y_test, df_test,
                    strategy=strategy,
                    confidence_threshold=threshold,
                    stake_pct=stake_pct,
                    initial_bankroll=float(initial_br),
                )

            # Métriques principales
            pnl = result["total_pnl"]
            roi = result["roi_pct"]
            pnl_color  = "normal" if pnl >= 0 else "inverse"
            roi_color  = "normal" if roi >= 0 else "inverse"

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💰 Bankroll finale",
                        f"{result['final_bankroll']:.0f} €",
                        f"{pnl:+.0f} €")
            col2.metric("📊 ROI", f"{roi:+.2f}%")
            col3.metric("🎯 Paris placés", result["bets_placed"])
            col4.metric("✅ Taux de réussite", f"{result['win_rate']}%")
            col5.metric("📉 Drawdown max", f"-{result['max_drawdown']:.0f} €")

            st.divider()

            # Graphiques
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(bankroll_chart(result["bankroll_history"], float(initial_br)),
                                use_container_width=True)
            with col2:
                st.plotly_chart(results_donut(result["bets_won"], result["bets_placed"]),
                                use_container_width=True)

            st.plotly_chart(pnl_bar_chart(result["details"]), use_container_width=True)

            # Détails des paris
            with st.expander("🔍 Détails des paris"):
                df_show = result["details"][result["details"]["stake"] > 0].copy()
                df_show["résultat"] = df_show.apply(
                    lambda r: "✅" if r["bet"] == r["actual"] else "❌", axis=1
                )
                st.dataframe(
                    df_show[["match", "actual", "bet", "odd", "stake", "pnl", "bankroll", "résultat"]],
                    use_container_width=True, height=300
                )

    # ======================================================================
    # TAB 3 — DONNÉES
    # ======================================================================
    with tab3:
        st.header("📊 Aperçu des données")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total matchs", len(df))
        col2.metric("Saisons",
                    f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}")
        col3.metric("Équipes", df["HomeTeam"].nunique())
        col4.metric("Features", X.shape[1])

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            dist = df["FTR"].value_counts()
            fig = px.pie(values=dist.values, names=["Domicile (H)", "Nul (D)", "Extérieur (A)"],
                         color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                         title="Distribution des résultats")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            goals_h = df.groupby(df["Date"].dt.year)["FTHG"].mean()
            goals_a = df.groupby(df["Date"].dt.year)["FTAG"].mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Domicile", x=goals_h.index.astype(str),
                                  y=goals_h.values, marker_color="#2ecc71"))
            fig.add_trace(go.Bar(name="Extérieur", x=goals_a.index.astype(str),
                                  y=goals_a.values, marker_color="#e74c3c"))
            fig.update_layout(
                title="Buts moyens par saison",
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                xaxis=dict(gridcolor="#222"),
                yaxis=dict(gridcolor="#222"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Derniers matchs")
        st.dataframe(
            df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].tail(20),
            use_container_width=True
        )


if __name__ == "__main__":
    main()
