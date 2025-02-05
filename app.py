from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Load data and train model
data1 = pd.read_csv("Project_train.csv")
x = data1[['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
           'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
           'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
           'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
           'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors']]
y = data1['FloodProbability']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

mse = mean_squared_error(y_test, model.predict(x_test))
r2 = r2_score(y_test, model.predict(x_test))

@app.route('/')
def home():
    return render_template('home.html', mse=mse, r2=r2)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    try:
        features = [float(request.form[feature]) for feature in x.columns]
        prediction = model.predict([features])[0]
        return render_template('home.html', prediction=prediction, mse=mse, r2=r2)
    except ValueError:
        return render_template('home.html', error="Please enter valid numeric values for all fields.", mse=mse, r2=r2)

if __name__ == '__main__':
    app.run(debug=True)
