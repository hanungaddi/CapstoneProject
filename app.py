import pandas as pd
import numpy as np
import json
import urllib.request

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import compose
from sklearn import impute

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app=app,
          version="1.0",
          title="Meal recommender API",
          description=" trained model that can recommend a meal")

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

recommender_namespace = api.namespace('food_recommender', description='a trained model that can recommend food using AKG Values (energi, protein, karbohidrat_total, lemak_total)')

@ recommender_namespace.route('/predict', methods=['POST'])
class Recommender(Resource):
    @ recommender_namespace.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'}, params={'food_name': {'description': 'food names', 'type': 'string', 'required': False}, 'when': {'description': 'when do you want the AI to recommend 1 (for breakfast), 2 (for lunch), 3 (for dinner). ex: 2, 3 can be 1 or more', 'type': 'string', 'required': False}})
    @ cross_origin()
    def get(self):
        # Filter out the request arguments
        args = request.get_json()
        food_name = args['food_name'] or None
        schedule = args['when'].split(', ') or None

        data = formulate(food_name, schedule)
        res = food_predict(data)

        return jsonify(res)

chatbot_namespace = api.namespace('smart_chatbot', description='a trained model for chatbot used by healthymeal app')

@ chatbot_namespace.route('/predict', methods=['POST'])
class Predict(Resource):
    @ recommender_namespace.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'}, params={'chat_content': {'description': 'chat contents', 'type': 'string', 'required': False}})
    @ cross_origin()
    def get(self):
        # Filter out the request arguments
        args = request.get_json()
        chat_content = args['chat_content'] or None

        res = chat_predict(chat_content)

        return jsonify(res)

def json_to_dataframe(data):
    # load the database into a dataframe
    datas = pd.read_json(data)
    # datas = pd.read_csv(database, delimiter=';')
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
    located_data = dataframecopy.loc[dataframecopy['food_name'] == food_name.lower()]
    food_values = [located_data['energi'].values[0],located_data['protein'].values[0],located_data['karbohidrat_total'].values[0],located_data['lemak_total'].values[0]]
    # find food_name in the database else [0, 0, 0, 0] (Consist of energi, protein, karbohidrat_total, lemak_total)

    # Formulate with AKGs (for now using Male 19-29 Years Old)
    AKG = np.array([2650, 65, 430, 75])
    AKG_left = AKG - np.array(food_values)
    # divide to a spesific values using the schedule
    breakfast_values = None
    lunch_values = None
    dinner_values = None
    snack_values = None
    for i in schedule:
        if int(i) == 1:
            breakfast_values = AKG_left * 3 / 18
            breakfast_values = breakfast_values.tolist()
        if int(i) == 2:
            lunch_values = AKG_left * 6 / 18
            lunch_values = lunch_values.tolist()
        if int(i) == 3:
            dinner_values = AKG_left * 4 / 18
            dinner_values = dinner_values.tolist()
        if int(i) == 4:
            snack_values = AKG_left * 1 / 18
            snack_values = snack_values.tolist()
    
    return [breakfast_values, lunch_values, dinner_values, snack_values]


def food_predict(data):
    result = {}

    for index,meal in enumerate(data):

        if meal != None:
            for list,i in enumerate(meal):
                meal[list] = float("%.1f" % i)

            # Transform the data to pandas dataframe
            test_data = preprocessor.transform(pd.DataFrame(data=[meal], index=np.arange(len([meal])), columns=['energi','protein','karbohidrat_total','lemak_total']))
            # Predict the data
            test_prediction = (-model.predict(test_data)).argsort()
            # Take 3 top data

            predicted = []
            for predicted_food in test_prediction[0][:5]:
                predicted.append({'nama_makanan': label_names[predicted_food], 'gizi':{'energi':dataframe.loc[predicted_food,:]['energi'],'protein':dataframe.loc[predicted_food,:]['protein'],'karbohidrat_total':dataframe.loc[predicted_food,:]['karbohidrat_total'],'lemak_total':dataframe.loc[predicted_food,:]['lemak_total']}})

            gizi_value = {'energi':meal[0],'protein':meal[1],'karbohidrat_total':meal[2],'lemak_total':meal[3]}

            if index == 0:
                result['breakfast'] = {'gizi_needed':gizi_value, 'recommended':predicted}
            if index == 1:
                result['lunch'] = {'gizi_needed':gizi_value, 'recommended':predicted}
            if index == 2:
                result['dinner'] = {'gizi_needed':gizi_value, 'recommended':predicted}
            if index == 3:
                result['snack'] = {'gizi_needed':gizi_value, 'recommended':predicted}
    
    return result

def chat_predict(chat_content):
    result = {}
    label = ['greeting','rekomendasi','random']

    token_list = tokenizer.texts_to_sequences([chat_content])[0]
    print(token_list)
	# Pad the sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')

    predicted = chatbot_model.predict(token_list, verbose=0)
    # Predict the label based on the maximum probability
    predicted = np.argmax(predicted, axis=-1).item()
    # if max(predicted[0]) >= 0.90 and max(predicted[0]) < 1.0:
	#     predicted = np.argmax(predicted, axis=-1).item()
    # else:
	#     predicted = -1

    result['predicted_label'] = label[predicted]

    return result

def get_all_food():
    url = "http://34.101.228.44:8080/food"
    req = urllib.request.Request(url, method='GET')
    req.add_header('Content-Type', 'application/json')
    returned_data = urllib.request.urlopen(req)
    result = returned_data.read()
    r = result.decode('utf-8')

    return r

if __name__ == '__main__':
    """ Food_Recommender """

    data = get_all_food()
    data_json = json.loads(data)

    # Load Database (?)
    dataframe = pd.json_normalize(data_json['data'])
    dataframe.pop('id')
    dataframe["food_name"] = dataframe["food_name"].str.lower()
    dataframecopy = dataframe
    label_names = dataframecopy['food_name'].to_list()

    # Preprocessing Data
    preprocessor = data_preprocessing(dataframe)
    preprocessor.fit_transform(dataframe)

    # Load Model
    model = load_model('./food_recommender_model/saved_model/testing_model_3.h5')

    """ Smart_Chatbot """
    CHATBOT_PATH = "./chatbot_model/saved_model/simple_chatbot_model/"

    # Load Tokenizer
    tokenizer_file = open(f'{CHATBOT_PATH}tokenizer.json', 'r')
    tokenizer_data = json.load(tokenizer_file)
    tokenizer = tokenizer_from_json(tokenizer_data)

    # Load max_sequence_length
    with open(f'{CHATBOT_PATH}max_sequence_length.txt', 'r') as file:
        max_sequence_len = int(file.read())
        file.close()
    
    # Load Model
    chatbot_model = load_model(f'{CHATBOT_PATH}first_model.h5')

    # Run the app
    app.secret_key = 'healthymealAPI2022'
    app.run(debug=True, host="0.0.0.0", port=8080)
    #app.run(host='127.0.0.1', port='5000', debug=True)

    
