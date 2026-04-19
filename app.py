from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model
try:
    pipe = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    raise SystemExit(
        "ERROR: model.pkl not found. "
        "Please run `python data_cleaning.py` then `python model.py` first."
    )
# ── IPL 2025 constants ────────────────────────────────────────────────────────
TOTAL_BALLS      = 120          # 20 overs × 6 balls
MAX_WICKETS      = 10           # innings ends at 10 wickets
POWERPLAY_END    = 6            # Powerplay: overs 1–6 (IPL / T20 rule)
MAX_BOWLER_OVERS = 4            # each bowler may bowl max 4 overs

VALID_TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]
# ─────────────────────────────────────────────────────────────────────────────


def parse_cricket_overs(overs_float: float) -> tuple[int, int]:
    """
    Convert cricket over notation (e.g. 12.4 → 12 complete overs, 4 balls)
    into (completed_overs, balls_in_current_over).

    IPL RULE: balls-in-over must be 0–5.
    Uses round() to avoid floating-point drift (12.3 − 12 = 0.2999…).
    """
    completed_overs     = int(overs_float)
    balls_in_this_over  = round((overs_float - completed_overs) * 10)
    return completed_overs, balls_in_this_over


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    error = None

    # ── 1. Parse raw form values ──────────────────────────────────────────────
    batting_team    = request.form['batting_team']
    bowling_team    = request.form['bowling_team']
    city            = request.form['city']
    current_score   = int(request.form['current_score'])
    overs           = float(request.form['overs'])
    wickets         = int(request.form['wickets'])          # wickets FALLEN
    runs_in_prev_5  = int(request.form['runs_in_prev_5'])

    # ── 2. IPL rule validations ───────────────────────────────────────────────

    # Teams must be different
    if batting_team == bowling_team:
        error = "Batting team and bowling team cannot be the same."
        return render_template('index.html', error=error)

    # Teams must be from the current IPL franchise list
    if batting_team not in VALID_TEAMS or bowling_team not in VALID_TEAMS:
        error = "Please select valid IPL 2025 franchises."
        return render_template('index.html', error=error)

    # Score cannot be negative
    if current_score < 0:
        error = "Current score cannot be negative."
        return render_template('index.html', error=error)

    # Parse cricket over notation
    completed_overs, balls_in_this_over = parse_cricket_overs(overs)

    # IPL RULE: balls-in-over is 0–5 (ball 6 closes the over → next over starts)
    if balls_in_this_over > 5:
        error = (
            f"Invalid over notation '{overs}': the decimal part represents "
            "balls bowled in the current over and must be .0–.5 "
            "(e.g. 12.5 = 12 overs and 5 balls)."
        )
        return render_template('index.html', error=error)

    # IPL RULE: T20 match is 20 overs (120 balls)
    balls_bowled = completed_overs * 6 + balls_in_this_over
    if balls_bowled <= 0:
        error = "At least one ball must have been bowled before predicting."
        return render_template('index.html', error=error)
    if balls_bowled >= TOTAL_BALLS:
        error = "The innings is already complete (20 overs played)."
        return render_template('index.html', error=error)

    # IPL RULE: maximum 10 wickets in an innings
    if wickets >= MAX_WICKETS:
        error = (
            f"Wickets fallen cannot be {wickets}. "
            "10 wickets ends the innings — prediction is not needed."
        )
        return render_template('index.html', error=error)
    if wickets < 0:
        error = "Wickets fallen cannot be negative."
        return render_template('index.html', error=error)

    # Runs in previous 5 overs must be non-negative
    if runs_in_prev_5 < 0:
        error = "Runs in the last 5 overs cannot be negative."
        return render_template('index.html', error=error)

    # Sanity: last-five runs cannot exceed current total
    if runs_in_prev_5 > current_score:
        error = "Runs in last 5 overs cannot exceed the current total score."
        return render_template('index.html', error=error)

    # ── 3. Feature engineering (must match data_cleaning.py exactly) ──────────

    balls_left   = TOTAL_BALLS - balls_bowled
    wickets_left = MAX_WICKETS - wickets

    # CRR formula: (runs × 6) / balls_bowled  — same formula used in training.
    # NOTE: 'overs' in cricket notation (12.4) ≠ 12.4 mathematical overs,
    #       so we MUST use balls_bowled here, not the raw overs float.
    crr = (current_score * 6) / balls_bowled

    # Powerplay flag (IPL rule: overs 1–6, only 2 fielders outside 30-yard circle)
    in_powerplay = completed_overs < POWERPLAY_END

    # ── 4. IPL phase features (must mirror model.py's add_ipl_features) ─────────
    # These are the same derived columns added during training.
    # Omitting them would cause a feature-count mismatch and wrong predictions.

    # Powerplay: overs 1–6 (IPL rule — only 2 fielders outside 30-yard circle)
    is_powerplay    = 1 if balls_bowled <= 36 else 0
    # Death overs: overs 17–20 (last 24 balls — free-hitting acceleration phase)
    is_death        = 1 if balls_left <= 24 else 0
    # Wicket pressure ratio: wickets available per ball remaining
    wicket_pressure = wickets_left / (balls_left + 1)

    # ── 5. Build the input DataFrame for the model ────────────────────────────
    input_df = pd.DataFrame({
        'batting_team'    : [batting_team],
        'bowling_team'    : [bowling_team],
        'city'            : [city],
        'current_score'   : [current_score],
        'balls_left'      : [balls_left],
        'wickets_left'    : [wickets_left],
        'crr'             : [crr],
        'last_five'       : [runs_in_prev_5],
        # IPL phase features — added in model.py training, required here too
        'is_powerplay'    : [is_powerplay],
        'is_death'        : [is_death],
        'balls_bowled'    : [balls_bowled],
        'wicket_pressure' : [wicket_pressure],
    })

    # ── 6. Predict ────────────────────────────────────────────────────────────
    result          = pipe.predict(input_df)
    predicted_score = int(result[0])

    # Ensure predicted score is at least the current score
    predicted_score = max(predicted_score, current_score)

    return render_template(
        'index.html',
        result          = predicted_score,
        current_score   = current_score,
        overs           = overs,
        in_powerplay    = in_powerplay,
        balls_left      = balls_left,
        wickets_left    = wickets_left,
        crr             = round(crr, 2),
    )


if __name__ == '__main__':
    app.run(debug=True)