from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# loading our saved model
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # grabbing all the values from the form
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = int(request.form['fuel_type'])
    seller_type = int(request.form['seller_type'])
    transmission = int(request.form['transmission'])
    owner = int(request.form['owner'])
    car_age = int(request.form['car_age'])

    # putting it all together for prediction
    features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age]])
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Estimated Car Price: ₹ {prediction:.2f} Lakhs')

if __name__ == '__main__':
    # Render (and most hosts) provide the port via the PORT env var
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
