import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pycaret.regression import load_model, predict_model
import openai
load_dotenv() 
import datetime

from langfuse.decorators import observe
from langfuse.openai import OpenAI

MODEL_NAME = 'h_marathon_model'
####'h_marathon_model' 'model_bieg'jestem Piotr, jestem mężczyzną, mam 60lat , czas na 5km to 40minut

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

model = get_model()

st.header("Przewidywanie czasu na mecie")

user_input = st.text_area('Przedstaw się nam! Podaj imię, płeć, wiek i czas na 5km')

openai.api_key = os.getenv("OPENAI_API_KEY")

@observe()
def extract_data(user_input):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Wyłuskaj dane z tekstu i zwróć JSON:
            {"Płeć_num": "1" dla mężczyzny lub "0" dla kobiety, "Wiek_w_latach": liczba, "Czas 5 km": "SS", "Imię": "imię"}
            Czas na 5km zawsze zwróć w postaci liczby sekund. Jeżeli użytkownik NIE NAPISAŁ wprost, że jest kobietą lub mężczyzną,
            zwróć "Płeć_num": null nie wnioskuj płci na podstawie imienia, Jeśli brakuje danych dla innych pól zwróć null dla danego pola."""},
            {"role": "user", "content": user_input}
],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)



if st.button('Oblicz czas'):
    data = extract_data(user_input)

    st.subheader(f"Witaj {data['Imię']}!")
        
    bledy = [k for k, v in data.items() if v is None]
    if bledy:
        st.error(f"Brakuje danych: {', '.join(bledy)}")
    else:    
        person_df = pd.DataFrame([{
            '5 km Czas': data['Czas 5 km'],
            'Płeć_num': data['Płeć_num'],
            'Wiek_w_latach': data['Wiek_w_latach']
        }])

        prediction = predict_model(model, data=person_df)['prediction_label'].values[0]
        Czas = str(datetime.timedelta(seconds=int(prediction)))
        st.markdown(f"<h2 style='color: blue; text-align: center;'>Przewidywany czas ukończenia Półmaratonu:<br><span style='color: gold;'>{Czas}</span><br>👍</h2>", unsafe_allow_html=True)