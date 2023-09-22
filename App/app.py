import streamlit as st
import pandas as pd

df = pd.read_csv('/Users/sergeidolin/anaconda3/Projects/Music_genre_classification/Data/kaggle_music_genre_train.csv')

st.write("""
# Прогнозирование жанров музыки!
         
         пслвслпылм
""")
         
st.sidebar.header('Параметры музыкального произведения')

def user_input():
    spech = st.sidebar.slider('Выразительность', df['speechiness'].dropna().min(), df['speechiness'].dropna().max())
    key = st.sidebar.slider('Базовый ключ (нота) произведения', st.sidebar.selectbox('Выберите основную ноту', (df['key'].unique())))
    data = {'Выразительность': spech,
            'Нота': key,}
    features = pd.DataFrame(data=data, index=[0])
    return features

f = user_input()
st.write(f)