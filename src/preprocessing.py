import pandas as pd

def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

def parse_datetime(df):
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'], utc=True).dt.tz_convert('Europe/Paris')
    return df

def extract_time_features(df):
    df['heure'] = df['date_et_heure_de_comptage'].dt.hour
    df['jour_mois'] = df['date_et_heure_de_comptage'].dt.day
    df['mois'] = df['date_et_heure_de_comptage'].dt.month
    df['annee'] = df['date_et_heure_de_comptage'].dt.year
    df['jour_semaine'] = df['date_et_heure_de_comptage'].dt.day_name()
    return df

def split_coordinates(df):
    df[['latitude', 'longitude']] = df['coordonnées_géographiques'].str.split(',', expand=True).astype(float)
    return df

def drop_columns(df):
    colonnes_a_supprimer = [
        'mois_annee_comptage', 'identifiant_du_site_de_comptage', 'identifiant_du_compteur',
        'nom_du_site_de_comptage', 'lien_vers_photo_du_site_de_comptage',
        'identifiant_technique_compteur', 'id_photos', "date_d'installation_du_site_de_comptage",
        'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1',
        'url_sites', 'type_dimage'
    ]
    return df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns])

def final_cleaning(df):
    df = df.dropna(subset=['latitude', 'longitude'])
    df.loc[:, 'nom_du_compteur'] = df['nom_du_compteur'].astype('category')
    df.loc[:, 'jour_semaine'] = df['jour_semaine'].astype('category')
    df.loc[:, 'latitude'] = df['latitude'].astype('float32')
    df.loc[:, 'longitude'] = df['longitude'].astype('float32')
    return df

def lister_compteurs(df):
    """
    Extrait la liste unique des compteurs avec leur nom, latitude et longitude.
    """
    if 'nom_du_compteur' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
        compteurs = (
            df[['nom_du_compteur', 'latitude', 'longitude']]
            .drop_duplicates()
            .sort_values('nom_du_compteur')
            .reset_index(drop=True)
        )
        return compteurs
    else:
        raise ValueError("Colonnes requises non présentes dans le DataFrame.")
    
def label_affluence(df):
    """
    Ajoute une colonne binaire 'affluence' (1 = affluence, 0 = non).
    Le seuil est basé sur la moyenne du comptage horaire pour chaque compteur.
    """
    df = df.copy()
    df['affluence'] = 0
    for compteur in df['nom_du_compteur'].unique():
        seuil = df[df['nom_du_compteur'] == compteur]['comptage_horaire'].mean()
        df.loc[
            (df['nom_du_compteur'] == compteur) & 
            (df['comptage_horaire'] > seuil),
            'affluence'
        ] = 1
    return df


