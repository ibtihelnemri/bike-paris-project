import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.sklearn

def prepare_features(df):
    X = df[['heure', 'jour_semaine', 'mois', 'latitude', 'longitude']]
    y = df['comptage_horaire']

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[['jour_semaine']]).toarray()

    X_final = np.hstack([X[['heure', 'mois', 'latitude', 'longitude']].values, X_encoded])
    return X_final, y, encoder

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f} | R²: {r2:.2f}")
    return model, mae, r2

def save_model(model, encoder, model_path="model.joblib", encoder_path="encoder.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

def run_pipeline(df):
    X, y, encoder = prepare_features(df)
    model, mae, r2 = train_and_evaluate(X, y)

    input_example = np.array([[8, 5, 48.8566, 2.3522] + [0]*len(encoder.categories_[0])])


    mlflow.set_experiment("Trafic Cycliste Paris")
    with mlflow.start_run():
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        #mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example
        )
        save_model(model, encoder)

    return model

def prepare_features_classification(df):
    X = df[['heure', 'jour_semaine', 'mois', 'nom_du_compteur']]
    y = df['affluence']
    
    # Encode categorical
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[['jour_semaine', 'nom_du_compteur']]).toarray()

    X_numeric = X[['heure', 'mois']].values
    X_final = np.hstack([X_numeric, X_encoded])
    return X_final, y, encoder

def train_and_evaluate_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    return clf

def run_classifier_pipeline(df):
    X, y, encoder = prepare_features_classification(df)
    model = train_and_evaluate_classifier(X, y)

    joblib.dump(model, "model_classifier.joblib")
    joblib.dump(encoder, "encoder_classifier.joblib")

    return model