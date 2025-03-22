from flask import Flask, render_template, request
import numpy as np
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))



app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    input_data = [float(x) for x in request.form.values()]
    input_data_np = np.asarray(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_np)[0]
    result = "The Person has Heart Disease" if prediction == 1 else "The Person does not have a Heart Disease"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
