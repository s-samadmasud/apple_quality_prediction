from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.svm import SVC

# load model===========================================
#svc = pickle.load(open('models/svc.pickle','rb'))

with open("models\\model.pkl", "rb") as file:
    # Load the data from the file
    loaded_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    size = request.form['size']
    weight = request.form['weight']
    sweetness = request.form['sweetness']
    crunchiness = request.form['crunchiness']
    juiciness = request.form['juiciness']
    ripeness = request.form['ripeness']
    acidity = request.form['acidity']

    # Create a NumPy array from user input (replace with error handling if needed)
    user_input = np.array([[float(size), float(weight), float(sweetness), float(crunchiness),
                           float(juiciness), float(ripeness), float(acidity)]])

    # Make prediction using the loaded model
    #prediction = svc.predict(user_input)[0]  # Assuming the model returns a single value

    prediction = loaded_model.predict(user_input)[0]

    # Format the prediction for display
    if prediction > 0.5:
        predicted_quality = "Good"
    else:
        predicted_quality = "Bad"

    return render_template('result.html', prediction=predicted_quality)

if __name__ == '__main__':
    app.run(debug=True)