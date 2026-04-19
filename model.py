import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv('cleaned_ipl_data.csv')
print(f"Loaded {len(df):,} rows.")

# ── 2. Data quality guards ─────────────────────────────────────────────────────
# Guard A: balls_left can never be negative in a T20 match (20 overs = 120 balls).
bad_balls = (df['balls_left'] < 0).sum()
if bad_balls:
    print(f"  Dropping {bad_balls} row(s) with balls_left < 0  (data corruption).")
    df = df[df['balls_left'] >= 0]

# Guard B: current_score can never exceed the final innings total.
# This was caused by a bug in data_cleaning.py (total calculated after dropna).
# Re-run data_cleaning.py first; this guard catches any lingering bad rows.
bad_score = (df['total'] < df['current_score']).sum()
if bad_score:
    print(f"  Dropping {bad_score} row(s) where total < current_score  (corrupt labels).")
    df = df[df['total'] >= df['current_score']]

print(f"  Clean dataset: {len(df):,} rows.\n")

# ── 3. IPL-informed feature engineering ───────────────────────────────────────
#
# IPL / T20 rules that shape scoring:
#
#   POWERPLAY (overs 1–6, balls 1–36):
#     Only 2 fielders allowed outside the 30-yard circle.
#     Teams attack early → higher boundary rate → run-rate spikes here.
#
#   MIDDLE OVERS (overs 7–15):
#     5 fielders allowed outside. Consolidation phase; economy tightens.
#
#   DEATH OVERS (overs 16–20, last 24 balls):
#     Free-hitting phase; teams accelerate heavily in the final 4 overs.
#
# A flat RandomForest has no notion of these phases from raw balls_left alone;
# making them explicit columns gives the model a direct signal.

def add_ipl_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    balls_bowled = 120 - d['balls_left']

    # Phase flags (IPL fielding restriction rules)
    d['is_powerplay']      = (balls_bowled <= 36).astype(int)  # overs 1-6
    d['is_death']          = (d['balls_left'] <= 24).astype(int)  # overs 17-20

    # balls_bowled as explicit feature (balls_left alone encodes the same info
    # but making both present helps tree splits find thresholds faster)
    d['balls_bowled']      = balls_bowled

    # Wicket pressure: how many wickets remain per ball left.
    # A team 8 down with 20 balls left is under far more pressure than
    # a team 2 down with 20 balls left — this ratio captures that directly.
    # +1 prevents division by zero on the very last ball.
    d['wicket_pressure']   = d['wickets_left'] / (d['balls_left'] + 1)

    return d


df = add_ipl_features(df)

print("Features after IPL engineering:")
print(" ", list(df.drop(columns=['total']).columns))
print()

# ── 4. Train / test split ──────────────────────────────────────────────────────
CATEGORICAL_COLS = ['batting_team', 'bowling_team', 'city']

X = df.drop(columns=['total'])
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Preprocessing + model pipeline ─────────────────────────────────────────
#
# OneHotEncoder with handle_unknown='ignore':
#   If a venue or team appears in production that wasn't in training data,
#   the encoder produces all-zero columns rather than crashing.
#
# RandomForestRegressor tuning notes vs the original:
#   n_estimators 100 → 200  : more trees → lower variance, negligible extra cost
#   max_features 'sqrt'     : default for regression; kept intentionally
#   min_samples_leaf 5      : prevents over-fitting on rare team/venue combos
#   n_jobs -1               : use all CPU cores

trf = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
     CATEGORICAL_COLS)
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('encoder', trf),
    ('model',   RandomForestRegressor(
                    n_estimators   = 200,
                    min_samples_leaf = 5,
                    n_jobs         = -1,
                    random_state   = 42,
                ))
])

print("Training model...")
pipe.fit(X_train, y_train)

# ── 6. Evaluation ──────────────────────────────────────────────────────────────
y_pred = pipe.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION (hold-out test set)")
print("="*50)
print(f"  R² Score : {r2:.4f}  (1.0 = perfect)")
print(f"  MAE      : {mae:.2f} runs")
print("="*50 + "\n")

# ── 7. Feature importances ─────────────────────────────────────────────────────
# Retrieve numeric feature names after one-hot encoding
ohe_feature_names = (
    pipe.named_steps['encoder']
    .named_transformers_['ohe']
    .get_feature_names_out(CATEGORICAL_COLS)
    .tolist()
)
passthrough_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
all_feature_names = ohe_feature_names + passthrough_cols

importances = pipe.named_steps['model'].feature_importances_
feat_imp = (
    pd.Series(importances, index=all_feature_names)
    .sort_values(ascending=False)
)

print("Top 15 feature importances:")
print(feat_imp.head(15).to_string())
print()

# ── 8. Save model ──────────────────────────────────────────────────────────────
pickle.dump(pipe, open('model.pkl', 'wb'))
print("model.pkl saved successfully!")
print()
print("IMPORTANT — app.py must compute these same features before calling pipe.predict():")
for col in passthrough_cols:
    if col not in ['current_score', 'balls_left', 'wickets_left', 'crr', 'last_five']:
        print(f"  • {col}")