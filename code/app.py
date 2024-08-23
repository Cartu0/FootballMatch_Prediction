from flask import Flask, render_template, request
import joblib
import numpy as np
from data_preprocessing import preprocess_data
from data_preprocessing import get_team_stats

app = Flask(__name__)

# Caricare il modello e le statistiche delle squadre
model = joblib.load('optimized_football_match_predictor.pkl')
_, _, _, _, team_stats = preprocess_data('processed_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        home_team = request.form['home_team'].strip().lower()
        away_team = request.form['away_team'].strip().lower()

        home_team_stats = get_team_stats(home_team, team_stats)
        away_team_stats = get_team_stats(away_team, team_stats)

        if home_team_stats is None or away_team_stats is None:
            prediction = "Una delle squadre non è presente nel dataset."
        else:
            prediction = predict_match_result(model, home_team_stats, away_team_stats)

        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

def predict_match_result(model, home_team_stats, away_team_stats):
    match_features = np.array([[
        home_team_stats['total_goals_for'], home_team_stats['avg_goals_for'], home_team_stats['total_goals_against'], home_team_stats['avg_goals_against'], home_team_stats['avg_speed'],
        away_team_stats['total_goals_for'], away_team_stats['avg_goals_for'], away_team_stats['total_goals_against'], away_team_stats['avg_goals_against'], away_team_stats['avg_speed']
    ]])

    probabilities = model.predict_proba(match_features)[0]

    result_encoded = model.predict(match_features)[0]
    if result_encoded == 1:
        result_symbol = '1'  # Vittoria squadra di casa
    elif result_encoded == 0:
        result_symbol = 'X'  # Pareggio
    else:
        result_symbol = '2'  # Vittoria squadra in trasferta

    return f"{result_symbol} (Probabilità: Casa {probabilities[2] * 100:.2f}%, Pareggio {probabilities[1] * 100:.2f}%, Ospite {probabilities[0] * 100:.2f}%)"

if __name__ == '__main__':
    app.run(debug=True)
