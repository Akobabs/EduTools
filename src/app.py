from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import shap
from src.data_preprocessing import preprocess_data
from src.models.ml_model import MLModel

app = Flask(__name__)

# Load model
rf_model = MLModel(model_type='rf').load('models/rf_model.joblib')
explainer = shap.TreeExplainer(rf_model.model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        input_data = {
            'gender': request.form['gender'],
            'region': request.form['region'],
            'highest_education': request.form['highest_education'],
            'age_band': request.form['age_band'],
            'num_of_prev_attempts': float(request.form['num_of_prev_attempts']),
            'studied_credits': float(request.form['studied_credits']),
            'total_clicks': float(request.form['total_clicks']),
            'avg_assessment_score': float(request.form['avg_assessment_score'])
        }
        
        # Preprocess input
        input_df = pd.DataFrame([input_data])
        # Assume preprocessing matches training (simplified for demo)
        input_df['gender'] = input_df['gender'].map({'M': 0, 'F': 1})
        input_df['region'] = input_df['region'].map({'North': 0, 'South': 1, 'East': 2, 'West': 3})
        input_df['highest_education'] = input_df['highest_education'].map({'A Level': 0, 'HE Qualification': 1, 'Lower Than A Level': 2})
        input_df['age_band'] = input_df['age_band'].map({'0-35': 0, '35-55': 1, '55+': 2})
        
        # Predict
        prediction = rf_model.predict(input_df)[0]
        prediction_label = 'Pass' if prediction == 1 else 'Fail'
        
        # SHAP explanation
        shap_values = explainer.shap_values(input_df)
        shap_data = {
            'features': input_df.columns.tolist(),
            'values': shap_values[1][0].tolist()
        }
        
        return render_template('results.html', prediction=prediction_label, shap_data=shap_data)
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df = preprocess_data(input_df)
    prediction = rf_model.predict(input_df)[0]
    shap_values = explainer.shap_values(input_df)
    return jsonify({
        'prediction': 'Pass' if prediction == 1 else 'Fail',
        'shap_values': shap_values[1][0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)