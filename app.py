
import pandas as pd
import numpy as np

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import compose
from sklearn import impute

from flask import Flask, jsonify, request, abort
from flask_cors import CORS, cross_origin
from pandas.io.json import json_normalize
from dateutil.parser import parse
from flask_restx import fields, Resource, Api, reqparse

app = Flask(__name__)
api = Api(app=app,
          version="1.0",
          title="Meal recommender API",
          description=" trained model that can recommend a meal")

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

predict_namespace = api.namespace('food_recommender', description='a trained model that can recommend food using AKG Values (Energi, Protein, Karbohidrat_total, Lemak_total)')

@ predict_namespace.route('/predict', methods=['GET'])
class Predict(Resource):
    @ predict_namespace.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'}, params={'food_name': {'description': 'food names', 'type': 'string', 'required': False}, 'when': {'description': 'when do you want the AI to recommend 1 (for breakfast), 2 (for lunch), 3 (for dinner). ex: 2, 3 can be 1 or more', 'type': 'string', 'required': False}})
    @ cross_origin()
    def get(self):
        # Filter
        parser = reqparse.RequestParser()
        parser.add_argument('food_name',  required=False, default=None)
        parser.add_argument('when', required=False, default=None)

        args = parser.parse_args()
        food_name = args['food_name'] or None
        schedule = args['when'].split(', ') or None

        data = formulate(food_name, schedule)
        res = predict(data)

        return jsonify(res)

def load(database):
    # load the database
    datas = pd.read_csv(database, delimiter=';')
    return datas

def data_preprocessing(dataframe):
    numeric_columns = dataframe.select_dtypes(include=['int64','float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),
        ('scaler', preprocessing.MinMaxScaler())
    ])
    preprocessor = compose.ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_columns),
    ])

    return preprocessor


def formulate(food_name,schedule):
    # Take values of food_name if there's any in database
    located_data = dataframecopy.loc[dataframecopy['Food_Name'] == food_name]
    food_values = [located_data['Energi'].values[0],located_data['Protein'].values[0],located_data['Karbohidrat_total'].values[0],located_data['Lemak_total'].values[0]]
    # find food_name in the database else [0, 0, 0, 0] (Consist of Energi, Protein, Karbohidrat_total, Lemak_total)

    # Formulate with AKGs (for now using Male 19-29 Years Old)
    AKG = np.array([2650, 65, 430, 75])
    AKG_left = AKG - np.array(food_values)
    # divide to a spesific values using the schedule
    breakfast_values = None
    lunch_values = None
    dinner_values = None
    for i in schedule:
        if int(i) == 1:
            breakfast_values = AKG_left * 1.5 / 6
            breakfast_values = breakfast_values.tolist()
        if int(i) == 2:
            lunch_values = AKG_left * 2.5 / 6
            lunch_values = lunch_values.tolist()
        if int(i) == 3:
            dinner_values = AKG_left * 2 / 6
            dinner_values = dinner_values.tolist()
    
    return [breakfast_values, lunch_values, dinner_values]


def predict(data):
    result = {}

    for index,meal in enumerate(data):

        if meal != None:
            # Transform the data to pandas dataframe
            test_data = preprocessor.transform(pd.DataFrame(data=[meal], index=np.arange(len([meal])), columns=['Energi','Protein','Karbohidrat_total','Lemak_total']))
            # Predict the data
            test_prediction = (-model.predict(test_data)).argsort()
            # Take 3 top data

            predicted = []
            for predicted_food in test_prediction[0][:3]:
                predicted.append({'nama_makanan': label_names[predicted_food], 'gizi':{'energi':dataframe.loc[predicted_food,:]['Energi'],'protein':dataframe.loc[predicted_food,:]['Protein'],'karbohidrat_total':dataframe.loc[predicted_food,:]['Karbohidrat_total'],'lemak_total':dataframe.loc[predicted_food,:]['Lemak_total']}})

            gizi_value = {'energi':meal[0],'protein':meal[1],'karbohidrat_total':meal[2],'lemak_total':meal[3]}

            if index == 0:
                result['breakfast'] = {'gizi_needed':gizi_value, 'recommended':predicted}
            if index == 1:
                result['lunch'] = {'gizi_needed':gizi_value, 'recommended':predicted}
            if index == 2:
                result['dinner'] = {'gizi_needed':gizi_value, 'recommended':predicted}
    
    return result

if __name__ == '__main__':
    # Load Database (?)
    dataframe = load('E:\BANGKIT2022\CapstoneProject\data_gathering\data\cleaned_data.csv')
    dataframecopy = dataframe
    label_names = dataframecopy['Food_Name'].to_list()

    # Preprocessing Data
    preprocessor = data_preprocessing(dataframe)
    preprocessor.fit_transform(dataframe)

    # Load Model
    model = load_model('./food_recommender_model/saved_model/testing_model_2.h5')

    # Run the app
    app.secret_key = 'foodpredictor2022'
    app.run(host='127.0.0.1', port='5000', debug=True)