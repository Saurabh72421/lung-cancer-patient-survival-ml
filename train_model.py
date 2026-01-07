from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
import joblib

X, y = preprocess_data('data/lung_cancer_data.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
joblib.dump(model, 'models/xgb_model.pkl')

print("Model trained and saved")