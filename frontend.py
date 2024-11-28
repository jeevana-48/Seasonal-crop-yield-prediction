from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Create a feature array for prediction
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        
        # Transform the features using the preprocessor
        transformed_features = preprocessor.transform(features)
        
        # Make predictions
        predicted_value = dtr.predict(transformed_features)[0]
        
        # Return the prediction result
        return render_template('index.html', predicted_value=predicted_value)

# Main entry point for the app
if __name__ == '__main__':
    app.run(debug=True)
