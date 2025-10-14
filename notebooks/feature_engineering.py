"""
Preprocess Paris Air Quality time-series data.
- Builds datetime index from (date, hour)
- Adds cyclical features (dayofweek sin/cos)
- Adds daily means (per calendar day) for pollutants
- Adds lockdown / curfew indicators with exact curfew hours
- Adds lag features (1h, 2h) for each pollutant
- Adds mean-by-hour features (global average at each hour of day)
"""


import numpy as np
import pandas as pd

POLLUTANTS = ['valeur_NO2', 'valeur_CO', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']

def _in_curfew_vec(d: pd.Series, h: pd.Series, start_date: str, end_date: str, start_hour: int, end_hour: int) -> pd.Series:
    """Masque vectorisé de couvre-feu (gère le passage de minuit)."""
    day_in = (d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))
    if start_hour < end_hour:
        hour_in = (h >= start_hour) & (h < end_hour)
    else:
        hour_in = (h >= start_hour) | (h < end_hour)
    return day_in & hour_in

def _add_lockdown_and_curfew(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute is_lockdown_paris et is_curfew_paris à partir de df['date'] et df['hour']."""
    d = pd.to_datetime(df['date']).dt.normalize()
    h = df['hour'].astype(int)

    # Confinements (nationaux → Paris inclus)
    lockdown = (
        ((d >= pd.Timestamp('2020-03-17')) & (d <= pd.Timestamp('2020-05-10'))) |
        ((d >= pd.Timestamp('2020-10-30')) & (d <= pd.Timestamp('2020-12-14'))) |
        ((d >= pd.Timestamp('2021-04-03')) & (d <= pd.Timestamp('2021-05-02')))
    )
    df['is_lockdown_paris'] = lockdown.astype(int)

    # Couvre-feux (horaires exacts)
    curfew = np.zeros(len(df), dtype=bool)
    # IDF & métropoles : 17–29 oct 2020, 21:00–06:00
    curfew |= _in_curfew_vec(d, h, '2020-10-17', '2020-10-29', 21, 6)
    # National : 15 déc 2020 – 15 jan 2021, 20:00–06:00
    curfew |= _in_curfew_vec(d, h, '2020-12-15', '2021-01-15', 20, 6)
    # Avancé à 18:00 : 16 jan – 18 mai 2021, 18:00–06:00
    curfew |= _in_curfew_vec(d, h, '2021-01-16', '2021-05-18', 18, 6)
    # 21:00–06:00 : 19 mai – 8 juin 2021
    curfew |= _in_curfew_vec(d, h, '2021-05-19', '2021-06-08', 21, 6)
    # 23:00–06:00 : 9 – 20 juin 2021 (levé le 20/06 matin)
    curfew |= _in_curfew_vec(d, h, '2021-06-09', '2021-06-20', 23, 6)

    # Exception : pas de couvre-feu la nuit du 24/12/2020
    exc = (
        ((d == pd.Timestamp('2020-12-24')) & (h >= 20)) |
        ((d == pd.Timestamp('2020-12-25')) & (h < 6))
    )
    curfew = curfew & ~exc

    df['is_curfew_paris'] = curfew.astype(int)
    return df

def preprocess_dataset(input_path: str = "/paris-air-quality-prediction/data/train.csv") -> pd.DataFrame:
    """
    Charge le CSV, construit l'index horaire, ajoute les features (dayofweek cyclique, daily means,
    lockdown/curfew, lags 1&2, mean_by_hour) et retourne le DataFrame final.
    """
    df = pd.read_csv(input_path)

    # Types de base
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['hour'] = df['hour'].astype(int)

    # Index datetime horaire
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    df = df.set_index('datetime').sort_index()
    df = df.asfreq('H')  # insère des NaN pour les heures manquantes

    # Reconstruire date/hour si besoin après asfreq
    if 'date' not in df.columns or df['date'].isna().any():
        df['date'] = df.index.normalize()
    if 'hour' not in df.columns or df['hour'].isna().any():
        df['hour'] = df.index.hour

    # Encodage cyclique du jour de la semaine
    df['dayofweek'] = df.index.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Moyenne par jour calendaire (même valeur pour les 24 lignes du jour)
    df['date_only'] = df.index.normalize()
    daily_means = df.groupby('date_only')[POLLUTANTS].transform('mean').add_suffix('_mean_24h')
    df = pd.concat([df, daily_means], axis=1)

    # Indicateurs confinement / couvre-feu
    df = _add_lockdown_and_curfew(df)

    # Lags 1h & 2h pour chaque polluant
    for col in POLLUTANTS:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)

    # Moyenne globale par heure (profil horaire historique)
    for col in POLLUTANTS:
        df[f'{col}_mean_by_hour'] = df.groupby('hour')[col].transform('mean')

    # Nettoyage colonne helper
    df = df.drop(columns=['date_only'])

    return df
