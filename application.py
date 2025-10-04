import pickle
import numpy as np
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('TEMPERATURE'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('WS'))
            Rain = float(request.form.get('RAIN'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('CLASSES'))
            Region = float(request.form.get('REGION'))

            new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            new_data_scaled = standard_scaler.transform(new_data)

            result = ridge_model.predict(new_data_scaled)[0]

            return render_template('home.html', results=result)

        except Exception as e:
            
            return render_template('home.html', results=f"Error: {e}")

    return render_template('home.html', results=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
