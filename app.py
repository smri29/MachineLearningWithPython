from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open(r'C:\Users\WALTON\Documents\GitHub\MachineLearningWithPython\Sessions\Day12\final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])  # 0 for female, 1 for male
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])  # 0 for no, 1 for yes
    region = request.form['region']  # Get the region as a string

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                               columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # One-hot encode the region
    input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)

    # Ensure the input data has the same columns as the model
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True) 