import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(path):
    df = pd.read_csv(path)

    df.drop('Patient_ID', axis=1, inplace=True)

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Survival_Status', axis=1)
    y = df['Survival_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y