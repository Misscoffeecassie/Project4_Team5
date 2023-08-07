import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Unpickling all models
unpickle_lr_model = pickle.load(open('models_pickle/lr_model.pkl','rb'))
unpickle_dt_model = pickle.load(open('models_pickle/dt_model.pkl','rb'))
unpickle_rf_model = pickle.load(open('models_pickle/rf_model.pkl','rb'))
unpickle_knn_model = pickle.load(open('models_pickle/knn_model.pkl','rb'))
unpickle_svc_model = pickle.load(open('models_pickle/svc_model.pkl','rb'))
unpickle_crop_info = pickle.load(open('models_pickle/crop_info.pkl','rb'))
unpickle_crop_name = pickle.load(open('models_pickle/crop_name.pkl','rb'))

content = {'Rice':'Rice is a warm-season crop that requires constant heat and humidity to grow. It can be cultivated as an annual in warm climates.',
            'Maize':"Maize crop can be cultivated on a very wide range of agro-climatic conditions.However, moderate temperature with plentiful supply of water is most favorable,and the development and yield of maize is found more in hot regions. ",
            'Chickpea':"Chickpeas are cool-season crops that prefer moderate temperatures for growth. They have a wide range of adaptability to different climates, but they generally thrive in specific temperature and rainfall conditions",
            'Kidney Bean':"Kidney-Ben grows well in tropical and temperate areas receiving 60 to 150 cm of rainfall annually. The main dried bean producers are from Myanmar, India, Brazil,China, the United States, Tanzania and Mexico.",
            'Pigeon Pea':"Pigeon pea is a tropical leguminous crop that thrives in warm and subtropical climates. ",
            'Moth Bean':"Moth bean is a warm-season legume crop that is primarily grown for its edible seed. It is adapted to arid and semi-arid regions and can tolerate drought and high temperatures.",
            'Mung Bean':"Mung bean are a warm-season legume crop that thrives in warm temperature, adequate sunlight, and well-distributed rainfall.",
            'Urad Bean':"Urad Bean grows best in hot and humid condition, cultivation is characterized by warm temperatures and suitable rainfall patterns.",
            'Lentil':"Lentils are a drought-tolerant, cool-season crop. They are usually grown in semi- arid climate without irrigation",
            'Pomegranate':"Pomegranate is widely grown in the subtropics and tropical, in colder climates it will often fail to fruit. They are well suited to regions with long, hot summers and mild winters.", 
            'Banana':"Bananas are tropical plants that require a warm and humid climate to thrive.",
            'Mango':"Mango will tolerate a wide range of climates, from warm temperate to tropical and is susceptible to cold.",
            'Grapes':"Generally grape cultivation varies depending in the intended use i.e., for fresh consumption and wine production. However, Grapes thrive in temperate to subtropical climates",
            'Watermelon':"Watermelons are sensitive to cold temperatures and thrives in warm and tropical to subtropical climates. They are produced in every mainland Australian state with majority of production in Queensland, New South Wales, Northern Territory and Western Australia.",
            'Muskmelon':"Muskmelon is a warm-season fruit crop that requires specific climate conditions for optimal growth and fruit development. It thrives in warm and sunny climates",
            'Apple':"Apple tends to thrive in climates where it’s cold in the winter, moderate in the summer and has medium to high humidity rather than a hot and dry climate.",
            'Orange':"Orange thrives in warm temperatures but can tolerate a range of climates within the subtropical and tropical zones.",
            'Papaya':"Papaya thrives best under warm, humid conditions. It’s a native to Central America and is grown in tropical and warmer subtropical areas worldwide.",
            'Coconut':"Coconut is a tropical palm, preferring humid tropical climate and thrives in specific climate condition",
            'Cotton':"Cotton is a tropical and subtropical crop that requires warm temperatures or its growth and development.",
            'Jute':"The suitable climate for growing jute is a warm and wet climate. It is a tropical plant that is also known as warm-season and thrives in warm temperatures.",
            'Coffee':"The predominant climate is tropical and equatorial, where humidity prevails throughout the year and is between 60% to 80%. Coffee is grown in a belt around the equator known as the “Coffee belt”."}

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    nitrogen = request.form['nitrogen']
    phosphorus = request.form['phosphorus']
    potassium = request.form['potassium']
    temp = request.form['temperature']
    humidity = request.form['humidity']
    ph = request.form['ph']
    rainfall = request.form['rainfall']

    # Convert user data into dict
    user_data =pd.DataFrame({'N':nitrogen,'P':phosphorus,'K':potassium,'temperature':temp,'humidity':humidity,'ph':ph,'rainfall':rainfall},index=[0])
    
    #Predcit the crop
    predicted_crop_lr = unpickle_lr_model.predict(user_data)
    predicted_crop_dt = unpickle_dt_model.predict(user_data)
    predicted_crop_rf = unpickle_rf_model.predict(user_data)
    predicted_crop_knn = unpickle_knn_model.predict(user_data)
    predicted_crop_svc = unpickle_svc_model.predict(user_data)
    for i in unpickle_crop_name:
        if predicted_crop_rf[0] == i:
            misc_info = unpickle_crop_info[i]
            img = os.path.join('static','image',i+".jpeg")
            para = content[i]
    result = {'crop':predicted_crop_rf[0],'N':nitrogen,'P':phosphorus,'K':potassium,'temp':temp,
              'humidity':humidity,'ph':ph,'rainfall':rainfall,'misc':misc_info,'img':img,'para':para}
    return render_template('prediction.html',result =result)

    
if __name__ =="__main__":
    app.run(debug=True)