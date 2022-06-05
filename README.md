# CapstoneProject

## How to run API
- install python3.8
- pip install -r requirement.txt
- run app.py

## How to use API
- Food Recomender
  - http://host:port/food_recommender/predict
  - Content-Type: "application/json"
  - Body: {"food_name": "food_name","when": "1, 2, 3, 4"}
  - 1 for Breakfast, 2 for Lunch, 3 for dinner, 4 for snack
 
- Smart Chatbot
  - http://host:port/smart_chatbot/predict
  - Content-Type: "application/json"
  - Body: {"chat_content": "chat_content"}
 
## Model

- Food Recomender
  - Tensorflow Model
  - Multi-Categorical classification
  - Trained with 700 food datas
  - Credits https://nilaigizi.com/ for the data

- Smart Chatbot
  - Tensorflow Model
  - Multi-Categorical classification
  - Trained with 3 label (greeting, rekomendasi, random) and specific chat_content in bahasa


## API

- Using flask as framework
- Preprocess user data with AKG
- Call the model to recommend 3 Best Food
- Smart chatbot model for classifying type of chat_content
