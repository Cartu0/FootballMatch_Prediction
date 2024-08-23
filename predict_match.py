import joblib
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data

def get_team_stats(team_name, team_stats):
    stats = team_stats[team_stats['team'] == team_name]
    if stats.empty:
        return None
    return stats.iloc[0]

def predict_match_result(model, home_team_stats, away_team_stats):
    match_features = np.array([[
        home_team_stats['total_goals_for'], home_team_stats['avg_goals_for'], home_team_stats['total_goals_against'], home_team_stats['avg_goals_against'], home_team_stats['avg_speed'],
        away_team_stats['total_goals_for'], away_team_stats['avg_goals_for'], away_team_stats['total_goals_against'], away_team_stats['avg_goals_against'], away_team_stats['avg_speed']
    ]])

    probabilities = model.predict_proba(match_features)[0]

    print(f"Probabilità di vittoria della squadra di casa: {probabilities[2] * 100:.2f}%")
    print(f"Probabilità di pareggio: {probabilities[1] * 100:.2f}%")
    print(f"Probabilità di vittoria della squadra in trasferta: {probabilities[0] * 100:.2f}%")

    result_encoded = model.predict(match_features)[0]

    # Convertire il risultato numerico in simbolo
    if result_encoded == 1:
        result_symbol = '1'  # Vittoria squadra di casa
    elif result_encoded == 0:
        result_symbol = 'X'  # Pareggio
    else:
        result_symbol = '2'  # Vittoria squadra in trasferta

    return result_symbol

if __name__ == "__main__":
    model = joblib.load('optimized_football_match_predictor.pkl')
    _, _, _, _, team_stats = preprocess_data('/Users/macbook/Desktop/progettino/processed_data.csv')

    home_team = input("Inserisci il nome della squadra di casa: ").strip().lower()
    away_team = input("Inserisci il nome della squadra in trasferta: ").strip().lower()

    home_team_stats = get_team_stats(home_team, team_stats)
    away_team_stats = get_team_stats(away_team, team_stats)

    if home_team_stats is None or away_team_stats is None:
        print("Una delle squadre non è presente nel dataset.")
    else:
        predicted_result = predict_match_result(model, home_team_stats, away_team_stats)
        print(f"La previsione per la partita è: {predicted_result}")
