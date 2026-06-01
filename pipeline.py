"""
⚽ Football Match Predictor — Pipeline Robuste
================================================
Données réelles : football-data.co.uk (format Premier League)
Modèles        : Logistic Regression, Random Forest, LightGBM
Cible          : FTR (H / D / A)

Usage :
    python pipeline.py
"""

import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIG — adapte le chemin à ton environnement
# ===========================================================================

DATA_DIR   = "/Users/herbstmathis/Documents/python/data"   # ← ton dossier
MODEL_DIR  = "/Users/herbstmathis/Documents/python/models"
WINDOW     = 5   # matchs glissants pour calculer la forme

# ===========================================================================
# 1. CHARGEMENT & NETTOYAGE
# ===========================================================================

def load_all_seasons(data_dir: str) -> pd.DataFrame:
    """
    Charge tous les fichiers de la forme 20182019.xlsx / .csv
    et les concatène dans un seul DataFrame trié par date.
    """
    patterns = [
        os.path.join(data_dir, "*.xlsx"),
        os.path.join(data_dir, "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans : {data_dir}")

    dfs = []
    for f in sorted(files):
        try:
            if f.endswith(".xlsx"):
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
            df["_source"] = os.path.basename(f)
            dfs.append(df)
            print(f"  ✅ {os.path.basename(f):25s} → {len(df):4d} matchs")
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(f)} ignoré : {e}")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage robuste :
    - Conversion de la date (format Excel numérique OU string)
    - Garde uniquement les colonnes utiles
    - Supprime les lignes sans résultat
    """
    # ── Date ──────────────────────────────────────────────────────────────
    if "Date" in df.columns:
        # Certaines saisons ont un entier Excel (ex: 45884), d'autres une string
        def parse_date(v):
            try:
                v = float(v)
                # Entier Excel → date Python
                return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))
            except (TypeError, ValueError):
                return pd.to_datetime(v, dayfirst=True, errors="coerce")

        df["Date"] = df["Date"].apply(parse_date)
    else:
        df["Date"] = pd.NaT

    # ── Colonnes obligatoires ─────────────────────────────────────────────
    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    df = df.dropna(subset=required)
    df = df[df["FTR"].isin(["H", "D", "A"])]

    # ── Colonnes optionnelles (complétées à 0 si absentes) ───────────────
    optional = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
                "B365H", "B365D", "B365A"]
    for col in optional:
        if col not in df.columns:
            df[col] = np.nan

    # ── Tri chronologique ────────────────────────────────────────────────
    df = df.sort_values("Date", na_position="first").reset_index(drop=True)

    print(f"\n📦 Dataset final : {len(df)} matchs | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================

def rolling_team_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Pour chaque match, calcule (AVANT ce match) les stats glissantes
    sur les `window` derniers matchs de chaque équipe.

    Features produites par équipe (home / away) :
      - form          : points moyens / 3  [0..1]
      - scored_avg    : buts marqués moyens
      - conceded_avg  : buts encaissés moyens
      - win_rate      : taux de victoires
      - shots_avg     : tirs moyens
      - shots_on_avg  : tirs cadrés moyens
    """
    df = df.copy()
    cols = [
        "home_form", "away_form",
        "home_scored",  "home_conceded",
        "away_scored",  "away_conceded",
        "home_win_rate","away_win_rate",
        "home_shots",   "away_shots",
        "home_shots_on","away_shots_on",
    ]
    for c in cols:
        df[c] = np.nan

    history: dict = {}   # team → liste de dicts

    for i, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        def stats(team):
            games = history.get(team, [])[-window:]
            if not games:
                return dict(form=0.45, scored=1.4, conceded=1.4,
                            win_rate=0.35, shots=11.0, shots_on=4.0)
            return dict(
                form      = np.mean([g["pts"]  for g in games]) / 3,
                scored    = np.mean([g["scored"]   for g in games]),
                conceded  = np.mean([g["conceded"] for g in games]),
                win_rate  = np.mean([g["win"]      for g in games]),
                shots     = np.mean([g["shots"]    for g in games]),
                shots_on  = np.mean([g["shots_on"] for g in games]),
            )

        hs, as_ = stats(home), stats(away)

        df.at[i, "home_form"]     = hs["form"]
        df.at[i, "away_form"]     = as_["form"]
        df.at[i, "home_scored"]   = hs["scored"]
        df.at[i, "home_conceded"] = hs["conceded"]
        df.at[i, "away_scored"]   = as_["scored"]
        df.at[i, "away_conceded"] = as_["conceded"]
        df.at[i, "home_win_rate"] = hs["win_rate"]
        df.at[i, "away_win_rate"] = as_["win_rate"]
        df.at[i, "home_shots"]    = hs["shots"]
        df.at[i, "away_shots"]    = as_["shots"]
        df.at[i, "home_shots_on"] = hs["shots_on"]
        df.at[i, "away_shots_on"] = as_["shots_on"]

        # ── Mise à jour historique ────────────────────────────────────────
        ftr = row["FTR"]
        history.setdefault(home, []).append({
            "pts":      3 if ftr == "H" else (1 if ftr == "D" else 0),
            "scored":   row["FTHG"], "conceded": row["FTAG"],
            "win":      int(ftr == "H"),
            "shots":    row["HS"]  if pd.notna(row["HS"])  else 11,
            "shots_on": row["HST"] if pd.notna(row["HST"]) else 4,
        })
        history.setdefault(away, []).append({
            "pts":      3 if ftr == "A" else (1 if ftr == "D" else 0),
            "scored":   row["FTAG"], "conceded": row["FTHG"],
            "win":      int(ftr == "A"),
            "shots":    row["AS"]  if pd.notna(row["AS"])  else 11,
            "shots_on": row["AST"] if pd.notna(row["AST"]) else 4,
        })

    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit X (features) et y (cible FTR).
    """
    df = rolling_team_stats(df)

    # Features dérivées
    df["form_diff"]    = df["home_form"]    - df["away_form"]
    df["scored_diff"]  = df["home_scored"]  - df["away_scored"]
    df["conceded_diff"]= df["home_conceded"]- df["away_conceded"]
    df["shots_diff"]   = df["home_shots"]   - df["away_shots"]
    df["win_rate_diff"]= df["home_win_rate"]- df["away_win_rate"]

    # Cotes bookmaker (excellente feature si disponible)
    use_odds = df["B365H"].notna().mean() > 0.5
    feature_cols = [
        "home_form",     "away_form",
        "home_scored",   "home_conceded",
        "away_scored",   "away_conceded",
        "home_win_rate", "away_win_rate",
        "home_shots",    "away_shots",
        "home_shots_on", "away_shots_on",
        "form_diff",     "scored_diff",
        "conceded_diff", "shots_diff",
        "win_rate_diff",
    ]
    if use_odds:
        feature_cols += ["B365H", "B365D", "B365A"]
        print("💰 Cotes bookmaker incluses dans les features")

    X = df[feature_cols].copy().fillna(0)
    y = df["FTR"]

    print(f"✅ Features : {X.shape[1]} variables | {X.shape[0]} matchs")
    print(f"   Distribution → H:{(y=='H').mean():.1%}  "
          f"D:{(y=='D').mean():.1%}  A:{(y=='A').mean():.1%}")
    return X, y


# ===========================================================================
# 3. ENTRAÎNEMENT
# ===========================================================================

def train(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Entraîne 3 modèles. Retourne un dict {nom: bundle}.
    """
    le = LabelEncoder().fit(["A", "D", "H"])

    models = {
        "Logistic Regression": {
            "model":  LogisticRegression(C=0.5, max_iter=2000, random_state=42),
            "scale":  True,
            "encode": False,
        },
        "Random Forest": {
            "model":  RandomForestClassifier(
                          n_estimators=300, max_depth=10,
                          min_samples_leaf=10, random_state=42, n_jobs=-1),
            "scale":  False,
            "encode": False,
        },
        "LightGBM": {
            "model":  lgb.LGBMClassifier(
                          n_estimators=300, learning_rate=0.05,
                          num_leaves=31, min_child_samples=20,
                          random_state=42, verbose=-1),
            "scale":  False,
            "encode": True,
        },
    }

    trained = {}
    for name, cfg in models.items():
        scaler = None
        X = X_train.copy()
        y = le.transform(y_train) if cfg["encode"] else y_train

        if cfg["scale"]:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        cfg["model"].fit(X, y)
        trained[name] = {
            "model":  cfg["model"],
            "scaler": scaler,
            "le":     le if cfg["encode"] else None,
            "classes": le.classes_ if cfg["encode"] else cfg["model"].classes_,
        }
        print(f"  ✅ {name}")

    return trained


# ===========================================================================
# 4. ÉVALUATION
# ===========================================================================

def evaluate(trained: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    results = []

    for name, b in trained.items():
        X = b["scaler"].transform(X_test) if b["scaler"] else X_test
        proba = b["model"].predict_proba(X)

        if b["le"] is not None:
            y_pred_enc = b["model"].predict(X)
            y_pred = b["le"].inverse_transform(y_pred_enc)
        else:
            y_pred = b["model"].predict(X)

        acc = accuracy_score(y_test, y_pred)
        ll  = log_loss(y_test, proba, labels=["A", "D", "H"])

        results.append({"Modèle": name, "Accuracy": round(acc, 4), "Log-Loss": round(ll, 4)})

        print(f"\n{'='*45}")
        print(f"📊 {name}  |  Accuracy {acc:.2%}  |  Log-Loss {ll:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Ext. (A)", "Nul (D)", "Dom. (H)"],
                                    zero_division=0))

    return pd.DataFrame(results).sort_values("Log-Loss").reset_index(drop=True)


# ===========================================================================
# 5. PRÉDICTION
# ===========================================================================

def predict(trained: dict, model_name: str, features: dict) -> dict:
    """
    Prédit les probabilités V/N/D pour un match.

    `features` est un dict avec les mêmes clés que les colonnes de X.
    Les clés manquantes sont complétées par des valeurs moyennes.
    """
    b = trained[model_name]

    defaults = {
        "home_form": 0.45, "away_form": 0.45,
        "home_scored": 1.4, "home_conceded": 1.4,
        "away_scored": 1.4, "away_conceded": 1.4,
        "home_win_rate": 0.35, "away_win_rate": 0.35,
        "home_shots": 11.0, "away_shots": 11.0,
        "home_shots_on": 4.0, "away_shots_on": 4.0,
        "form_diff": 0.0, "scored_diff": 0.0,
        "conceded_diff": 0.0, "shots_diff": 0.0,
        "win_rate_diff": 0.0,
        "B365H": np.nan, "B365D": np.nan, "B365A": np.nan,
    }
    row = {**defaults, **features}

    # Garde uniquement les colonnes sur lesquelles le modèle a été entraîné
    fn = getattr(b["model"], "feature_name_", None)
    if callable(fn):
        expected_cols = fn()
    elif isinstance(fn, list):
        expected_cols = fn
    else:
        expected_cols = list(defaults.keys())
    X = pd.DataFrame([{c: row.get(c, 0) for c in expected_cols}]).fillna(0)

    if b["scaler"]:
        X = b["scaler"].transform(X)

    proba = b["model"].predict_proba(X)[0]
    classes = b["classes"]

    result = dict(zip(classes, proba))
    label  = {"H": "🏠 Victoire domicile", "D": "🤝 Nul", "A": "✈️  Victoire extérieure"}

    print(f"\n⚽ [{model_name}]")
    for k in ["H", "D", "A"]:
        bar = "█" * int(result.get(k, 0) * 30)
        print(f"   {label[k]:<25} {result.get(k, 0):5.1%}  {bar}")

    winner = max(result, key=result.get)
    print(f"\n   → Pronostic : {label[winner]}  ({result[winner]:.1%})")
    return {label[k]: round(result.get(k, 0), 4) for k in ["H", "D", "A"]}


# ===========================================================================
# 6. SAUVEGARDE / CHARGEMENT
# ===========================================================================

def save(trained: dict, path: str = None):
    path = path or os.path.join(MODEL_DIR, "football_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(trained, path)
    print(f"\n💾 Modèle sauvegardé → {path}")


def load(path: str = None) -> dict:
    path = path or os.path.join(MODEL_DIR, "football_model.pkl")
    return joblib.load(path)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("⚽  FOOTBALL ML — Pipeline Robuste")
    print("=" * 55)

    # 1. Chargement
    print("\n--- 1. Chargement des données ---")
    raw = load_all_seasons(DATA_DIR)

    # 2. Nettoyage
    print("\n--- 2. Nettoyage ---")
    df = clean(raw)

    # 3. Features
    print("\n--- 3. Feature Engineering ---")
    X, y = build_features(df)

    # 4. Split — SANS shuffle pour respecter la chronologie
    #    80% train / 20% test (les matchs les plus récents = test)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"\n   Train : {len(X_train)} matchs | Test : {len(X_test)} matchs")

    # 5. Entraînement
    print("\n--- 4. Entraînement ---")
    trained = train(X_train, y_train)

    # 6. Évaluation
    print("\n--- 5. Évaluation ---")
    results = evaluate(trained, X_test, y_test)
    print("\n📋 Classement des modèles :")
    print(results.to_string(index=False))

    # 7. Exemple de prédiction : Arsenal (dom.) vs Man City (ext.)
    print("\n--- 6. Prédiction : Arsenal vs Man City ---")
    predict(
        trained, "LightGBM",
        features={
            "home_form":      0.80,   # Arsenal en très bonne forme
            "away_form":      0.75,   # Man City solide
            "home_scored":    2.4,
            "home_conceded":  0.8,
            "away_scored":    2.6,
            "away_conceded":  0.7,
            "home_win_rate":  0.65,
            "away_win_rate":  0.60,
            "home_shots":     16.0,
            "away_shots":     15.0,
            "home_shots_on":  6.0,
            "away_shots_on":  5.5,
            # Cotes bookmaker si disponibles :
            # "B365H": 2.20, "B365D": 3.40, "B365A": 3.10,
        }
    )

    # 8. Sauvegarde
    save(trained)
    print("\n✅ Pipeline complète — succès !")
