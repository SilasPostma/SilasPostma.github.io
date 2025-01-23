from flask import Flask, render_template, request
from flask_cors import CORS  # Importing CORS
import pandas as pd
import joblib
import numpy as np
import math

app = Flask(__name__)

CORS(app)

# Load model and preprocessor
model = joblib.load('final_random_forest.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Updated user_choices with all possible values
user_choices = {
    'geslacht': ['M', 'V'],
    'gewichtsklasse': ['L', 'Z'],
    'ervaring': ['1', '0'],
    'ploeg': ['MGH', 'EJZ', 'EJL', 'EJD', 'MGLD', 'MJD', 'EJH'],
    'zone': ['AT', 'ED', 'I', 'ID', 'ED+'],
    'spm': ['vrij', '26', '20', '24-26-28-30', '26-28-30-32', '28-30-32', '30-32-34', '22-24-26-28', 
            '24', '26-28', 'T28', '32', '28', '30-32', '22', '20-24', '20-22-24-26', '30', 
            '28-30-32-34', '28-30', '16', '22-24', '22-26', '23', '25', '27', '40'],
    'trainingype': ["5x5'", "3x15'", "3x20'", "3x2000m/5'r", '6000m', '4x1500m', "3x1000m/5'r", 
                    "6x500m/2'r", '1500m', "1'", '2000m', "30'", '3x2000m', "3x20'/3'r", "3x6x1'/1'r",
                    'minuutjes', '2x2000m', '6x500m', '3x1000m', "8x5'/3'r", "3x4000m/4'r", "4x8'/4'r",
                    "3x10'", '1000m', '1500m + 500m', "4x8'", "8x3'", '4x750m', "3x10'/5'r",
                    '2x2000m + 500m', "3x7x1'/1'r /3'r", '1000m + 500m', "3x5x1'/1'r /5'r", "4x5'",
                    "3x12'", "2x25'", '8x500m', '100m', '500m', "3x20'/5'r", '1500m+500m', "HOP+3x1'r",
                    '3x4000m', "3x7'", "3x8'", '1000m+500m', "3x10'/3'r", "3x2000/3'r", "4x5'/5'r",
                    '2000', "3x12'/3'r", "4x500/5'r", "3x3000/5'r", "3x15'/3'r", "8x3'/3'r",
                    "7x3'/3'r", "2x19'/3'r", "2x3000/5'r", "6x750/5'r", "2x2000/5'r", '1000',
                    "3x8'/5'r", '1500m+750m', "3x12'/5'r", "2x6x1'/1'r", "2x7x1'/1'r", "3x11'",
                    "6x6'", "3x13'", "5x8'", "20'", "3x2000m/4'r", "3x10'/4'r", "3x1000m/3'r",
                    "8x3'/2'r", "4x2000/5'r", "6x750/3'r", "3x7x1'/1'r", "3x1500/5'r", "4x8'/5'r",
                    '6000', "2x10'/5'r", "2x4x20''/40''r", "1x20'", "3x5x1'/1'r", "6x5'/2'r",
                    "7x3'/2'r", "3x2000/5'r", "4x4x40''/40''r", "9x3'/2'r", "3x8x1'/1'r",
                    "5x5'/3'r", "5x7'/3'r", "5x9'/3'r", "3x1000/3'r", "2x12'/8'r", "3x8x40'/20'r",
                    "3x16'/3'r", "3x7'/5'r", "6x3'/3'r", "9x3'/3'r", "2x10'/3'r", "2x8x1'/2'r",
                    "3x5x1'/2'r", "3x5x1'/1'30'r", '2x2000', '1x1500', '1x2000', "3x5x1'/1'30''r",
                    "5 x 20''", '3x1500', '1 x Step 2000', "4 x 8x30''"],
    'interval_tijd': ['300', '900', '1200', '60', '1800', '6x60', '480', '180', '600', '7x60/60r',
                      '5x60', '720', '1500', 'xx60', '3600', '420', '5x60/60r', '1140', '60/60',
                      '660', '360', '790', '7x60', '4x20', '7x1', '4x40', '8x60', '540', 'xx40',
                      '960', 'xx480', "10'", "8x1'", "5x1'", "20''", "30''"],
    'interval_nummer': ['avg', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'aantal_intervallen': ['x', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'interval_afstand': ['100.0', '500.0', '750.0', '1000.0', '1500.0', '2000.0', '3000.0', '4000.0', '6000.0'],
    'intervaltype': ['tijd', 'afstand'],
    'rust': ['0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '10.0', '60.0', '120.0', '180.0', '240.0', '300.0'],
    '500_split': ['']
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract the selected values from the form
        aantal_intervallen = request.form.get('aantal_intervallen')
        
        # Create base user input dictionary
        user_input = {
            'geslacht': request.form.get('geslacht'),
            'gewichtsklasse': request.form.get('gewichtsklasse'),
            'ervaring': int(request.form.get('ervaring')),
            'ploeg': request.form.get('ploeg'),
            'zone': request.form.get('zone'),
            'spm': request.form.get('spm'),
            'trainingype': request.form.get('trainingype'),
            'interval_tijd': int(request.form.get('interval_tijd')),
            'aantal_intervallen': aantal_intervallen,
            'interval_afstand': float(request.form.get('interval_afstand')),
            'intervaltype': request.form.get('intervaltype'),
            'rust': float(request.form.get('rust')),
        }

        # Get all predictions for each interval
        predictions = []
        for i in range(1, int(aantal_intervallen) + 1 if aantal_intervallen != 'x' else 1):
            split_key = f'500_split_{i}'
            interval_input = user_input.copy()
            interval_input['interval_nummer'] = str(i)
            interval_input['500_split'] = float(request.form.get(split_key) or 0)

            # Convert to DataFrame and preprocess
            input_df = pd.DataFrame([interval_input])
            
            categorical_cols = ['geslacht', 'gewichtsklasse', 'intervaltype', 'ploeg', 'zone', 
                              'spm', 'trainingype', 'interval_tijd', 'interval_nummer', 'aantal_intervallen']
            numerical_cols = ['500_split', 'ervaring', 'interval_afstand', 'rust']

            for col in categorical_cols:
                input_df[col] = input_df[col].astype(str)

            for col in numerical_cols:
                input_df[col] = input_df[col].astype(float)

            # Ensure consistent column names and data types
            missing_cols = set(preprocessor.feature_names_in_) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = np.nan

            # Preprocess the input data
            input_processed = preprocessor.transform(input_df)

            # Predict using the model
            time = model.predict(input_processed) * 4  # Scaling factor for prediction in seconds
            time = time.item()

            # Convert to minutes and seconds
            minutes = math.floor(time // 60)
            remaining_seconds = time % 60
            formatted_time = f"{minutes}:{remaining_seconds:05.2f}"
            
            predictions.append({
                'interval': i,
                'formatted_time': formatted_time,
                'total_seconds': time
            })

        # Calculate average time if there are multiple predictions
        if predictions:
            avg_seconds = sum(p['total_seconds'] for p in predictions) / len(predictions)
            avg_minutes = math.floor(avg_seconds // 60)
            avg_remaining_seconds = avg_seconds % 60
            avg_formatted_time = f"{avg_minutes}:{avg_remaining_seconds:05.2f}"
        else:
            avg_formatted_time = "No predictions"

        return render_template('result.html', 
                            predictions=predictions, 
                            average_time=avg_formatted_time)

    return render_template('index.html', user_choices=user_choices)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
