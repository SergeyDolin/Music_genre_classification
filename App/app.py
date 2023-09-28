import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from catboost import CatBoostClassifier
import requests
from urllib.parse import urlencode
import py7zr
import os

# import data
df = pd.read_csv('./App/train.csv')
music_recomend = pd.read_csv('../Data/music_genre.csv')

# download model from ya.disk
if os.path.isfile('../App/CatModel.dump.7z') is not True:
    
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/pdO-LvxKOcxZ3w'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open('../App/CatModel.dump.7z', 'wb') as f:   
        f.write(download_response.content)

# extract 7z with model
if os.path.isfile('../App/CatModel.dump') is not True \
    and os.path.isfile('../App/CatModel.dump.7z') is True:
    with py7zr.SevenZipFile('../App/CatModel.dump.7z', mode='r') as z:
        z.extractall()

# load model
cat_model = CatBoostClassifier()
cat_model.load_model('../App/CatModel.dump')


key_dict = {'N': -1,'C' : 1, 'C#' : 2, 'D' : 3, 'D#' : 4, 'E' : 5, 'F' : 6, 
        'F#' : 7, 'G' : 9, 'G#' : 10, 'A' : 11, 'A#' : 12, 'B' : 12}
mode_dict = {'Major' : 1, 'Minor' : 0, 'N': -1}
duration = {'Long' : 1, 'Normal' : 0, 'Short': -1}

# user_input -> get features from user
def user_input():
    spech = st.sidebar.slider('Выразительность', df['speechiness'].dropna().min(), df['speechiness'].dropna().max())
    key = st.sidebar.selectbox('Выберите основную ноту', key_dict.keys())
    inst = st.sidebar.slider('Инструментальность', df['instrumentalness'].min(), df['instrumentalness'].max())
    live = st.sidebar.slider('Привлекательность', df['liveness'].min(), df['liveness'].max())
    mode = st.sidebar.selectbox('Модальность трека', mode_dict.keys())
    tempo = st.sidebar.slider('Темп (BPM)', df['tempo'].min(), df['tempo'].max())
    dur = st.sidebar.selectbox('Длительность трека', duration.keys())
    data = {'speechiness': spech,
            'key': key_dict[key],
            'instrumentalness': inst,
            'liveness': live,
            'mode': mode_dict[mode],
            'tempo': tempo,
            'duration_class' : duration[dur],
            }
    features = pd.DataFrame(data=data, index=[0])
    return features


# 1
st.title("""
Прогнозирование жанров музыки!

         ...и по этому все так, произошло!
""")
st.sidebar.header('Параметры музыкального произведения')

f = user_input()

# 2
st.title('Признаки')
st.write(f)

# Features hist
options1 = {
"xAxis": {
    "type": "category",
    "data": ['Выразительность','Основная нота', 'Инструментальность', 'Привлекательность', 'Модальность', 'Темп', 'Длительность трека'],
},
"yAxis": {"type": "value"},
"series": [
    {"data": [f['speechiness'][0], f['key'][0]/10, f['instrumentalness'][0],f['liveness'][0],
                f['mode'][0]/10, f['tempo'][0]/200, f['duration_class'][0]/10], "type": "bar"}
],
}
st_echarts(options=options1)

# Predict with music recomendation
if st.button('Что за жанр?'):
    st.write('Думаю...🤨')
    cat_pred = cat_model.predict(f)
    st.write('Скорее всего это 🤔:', *(cat_pred))
    st.write('Предлагаю послушать топ-5 треков этого жанра по мнению слушателей 😎:', music_recomend.loc[(music_recomend['music_genre'] == cat_model.classes_[cat_pred[0].argmax()]), 'track_name'].head(5).values)
