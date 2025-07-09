import pandas as pd
from src import preprocessing as prep
from src import train_model as tm

# Chemin vers le fichier de données
DATA_PATH = "data/comptage-velo-donnees-compteurs.csv"

def main():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH, sep=';')

    print("Nettoyage des données...")
    df = prep.clean_columns(df)
    df = prep.parse_datetime(df)
    df = prep.extract_time_features(df)
    df = prep.split_coordinates(df)
    df = prep.drop_columns(df)
    df = prep.final_cleaning(df)
    compteurs = prep.lister_compteurs(df)
    compteurs.to_csv("data/liste_compteurs.csv", index=False)

    print("Entraînement du modèle...")
    tm.run_pipeline(df)

    print("Ajout de la colonne d'affluence...")
    df_affluence = prep.label_affluence(df)

    print("Entraînement du modèle de classification...")
    tm.run_classifier_pipeline(df_affluence)

    print("Modèles entraînés et sauvegardés avec succès.")

if __name__ == "__main__":
    main()

