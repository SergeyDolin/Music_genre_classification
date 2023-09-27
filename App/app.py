import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from catboost import CatBoostClassifier

df = pd.read_csv('/Users/sergeidolin/anaconda3/Projects/Music_genre_classification/App/train.csv')
raw_data = pd.read_csv('/Users/sergeidolin/anaconda3/Projects/Music_genre_classification/Data/kaggle_music_genre_train.csv')
num_features = raw_data.select_dtypes(exclude='object').columns.to_list()
key_dict = {'N': -1,'C' : 1, 'C#' : 2, 'D' : 3, 'D#' : 4, 'E' : 5, 'F' : 6, 
        'F#' : 7, 'G' : 9, 'G#' : 10, 'A' : 11, 'A#' : 12, 'B' : 12}
mode_dict = {'Major' : 1, 'Minor' : 0, 'N': -1}
duration = {'Long' : 1, 'Normal' : 0, 'Short': -1}
music_recomend = pd.read_csv('/Users/sergeidolin/anaconda3/Projects/Music_genre_classification/Data/music_genre.csv')
cat_model = CatBoostClassifier()
cat_model.load_model('/Users/sergeidolin/anaconda3/Projects/Music_genre_classification/App/CatModel.dump')

st.title("""
Прогнозирование жанров музыки!

         ...и по этому все так, произошло!
""")


st.sidebar.header('Параметры музыкального произведения')

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

f = user_input()


st.title('Признаки')
st.write(f)
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




if st.button('Что за жанр?'):
    st.write('Думаю...🤨')
    cat_pred = cat_model.predict(f)
    st.write('Скорее всего это 🤔:', *(cat_pred))
    st.write('Предлагаю послушать топ-5 треков этого жанра по мнению слушателей 😎:', music_recomend.loc[(music_recomend['music_genre'] == cat_model.classes_[cat_pred[0].argmax()]), 'track_name'].head(5))
