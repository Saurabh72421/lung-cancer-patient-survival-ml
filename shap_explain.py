import shap
import joblib
from preprocess import preprocess_data

X, y = preprocess_data('data/lung_cancer_data.csv')
model = joblib.load('models/xgb_model.pkl')

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.bar(shap_values)
shap.plots.waterfall(shap_values[0])