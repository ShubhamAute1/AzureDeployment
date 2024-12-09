import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# Load the trained model
with open('randomForestRegressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle predictions based on user input from a form.
    """
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        return render_template('home.html', prediction_text=f"AQI for Jaipur: {prediction[0]:.2f}")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Handle API predictions through JSON requests.
    """
    try:
        data = request.get_json(force=True)
        features = np.array(list(data.values()))
        prediction = model.predict([features])
        return jsonify(prediction[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
