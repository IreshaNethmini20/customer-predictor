from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here') # Change this to a secure random key in production

# Load the CatBoost model
try:
    model_path = 'website/model/catboost_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define satisfaction mapping
satisfaction_mapping = {
    0: 'High',
    1: 'Low',
    2: 'Medium'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/reports')
def reports():
    if 'prediction' not in session:
        # Redirect to analytics if no prediction
        return redirect(url_for('analytics'))
    return render_template('reports.html', prediction=session['prediction'])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please check server configuration.'}), 500

    try:
        data = request.get_json()
        input_data = {
            'Age': float(data.get('age', 0)),
            'Gender': data.get('gender', ''),
            'Occupation': data.get('occupation', ''),
            'Travel_Class': data.get('travel_class', ''),
            'State_of_Residence': data.get('state_of_residence', ''),
            'Duration_of_Stay_(Days)': float(data.get('duration_of_stays', 0)),
            'Number_of_Companions': float(data.get('number_of_companions', 0)),
            'Purpose_of_Travel': data.get('purpose_of_travel', ''),
            'Special_Requests': ', '.join(data.get('special_request', [])) if data.get('special_request') else 'No',
            'Loyalty_Program_Member': data.get('loyalty_program_member', ''),
            'Total_Price': float(data.get('total_price', 0)),
            'Destination_City': data.get('destination_city', ''),
            'Destination_Country': data.get('destination_country', ''),
            'Days_Before_Travel': float(data.get('days_before_travel', 0))
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        satisfaction_class = satisfaction_mapping.get(int(prediction), 'Unknown')

        score_map = {
            'High': 4.5,
            'Medium': 3.0,
            'Low': 1.5,
            'Unknown': 3.0
        }
        color_map = {
            'High': 'success',
            'Medium': 'info',
            'Low': 'danger',
            'Unknown': 'warning'
        }

        result = {
            'success': True,
            'satisfaction_score': round(score_map.get(satisfaction_class, 3.0), 2),
            'satisfaction_class': satisfaction_class,
            'color': color_map.get(satisfaction_class, 'warning'),
            'input_data': data  # Store full input for report
        }

        # Store in session for reports
        session['prediction'] = result

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/clear_prediction', methods=['POST'])
def clear_prediction():
    session.pop('prediction', None)
    return jsonify({'success': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)