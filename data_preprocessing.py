import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    # Caricare il dataset
    data = pd.read_csv(filepath)

    # Normalizzare i nomi delle squadre (rimuovere spazi extra e convertire in minuscolo)
    data['home_team'] = data['home_team'].str.strip().str.lower()
    data['away_team'] = data['away_team'].str.strip().str.lower()

    # Aggregare le statistiche per ciascuna squadra
    team_stats = data.groupby('home_team').agg({
        'home_team_goal': ['sum', 'mean'],
        'away_team_goal': ['sum', 'mean'],
        'home_speed': 'mean'
    }).reset_index()
    team_stats.columns = ['team', 'total_goals_for', 'avg_goals_for', 'total_goals_against', 'avg_goals_against', 'avg_speed']

    # Creare nuove funzionalità per ciascuna partita
    data = data.merge(team_stats, left_on='home_team', right_on='team', how='left')
    data = data.merge(team_stats, left_on='away_team', right_on='team', how='left', suffixes=('_home', '_away'))

    # Caratteristiche per l'addestramento
    X = data[['total_goals_for_home', 'avg_goals_for_home', 'total_goals_against_home', 'avg_goals_against_home', 'avg_speed_home',
              'total_goals_for_away', 'avg_goals_for_away', 'total_goals_against_away', 'avg_goals_against_away', 'avg_speed_away']]
    y = data['result']

    # Dividere il dataset in set di allenamento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, team_stats

def get_team_stats(team_name, team_stats):
    """
    Estrae le statistiche di una squadra specifica dal dataset aggregato.
    """
    stats = team_stats[team_stats['team'] == team_name]
    if stats.empty:
        return None
    return stats.iloc[0]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, team_stats = preprocess_data('/Users/macbook/Desktop/progettino/processed_data.csv')
    print(X_train.head())
    print(y_train.head())
    print(team_stats.head())
