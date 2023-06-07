import sys
import streamlit as st


# Imports and loads on google colab
if 'google.colab' in sys.modules:

    # Connect Google Colab to Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    path = '/content/drive/MyDrive/Colab Notebooks/PMDM/Project/Datasets/'

    from streamlit_jupyter import StreamlitPatcher
    sp = StreamlitPatcher()
    sp.jupyter()  # register patcher with streamlit

else:

    path = "./Datasets/"

# Regular Imports
import pandas as pd
import numpy as np

st.title("Model Analysis")

df = pd.read_csv("./model_performance_records.csv")
df['F1'] = df['F1'].astype(float)
df['F1_val'] = df['F1_val'].astype(float)
st.subheader("Model Performance Records")
st.dataframe(df)


st.write("----")
# Best model

st.header("Best Performing Model")
df_best = df[df['F1'] == df['F1'].max()]
st.subheader(df_best['Model'].iloc[0])


col_best_1, col_best_2 = st.columns(2)
with col_best_1:
    st.metric('F1', df_best['F1'].iloc[0])
with col_best_2:
    st.metric('F1_val', df_best['F1_val'].iloc[0])


st.write("Best Performing Model Parameters")
df_best_params = df_best.drop(['Date', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Accuracy_val', 'Precision_val', 'Recall_val', 'F1_val', 'Number of Outcome = 1','Number of Outcome = 0'], axis=1).T
df_best_params.dropna(inplace=True)
st.write(df_best_params.T.iloc[0])

st.write("----")
st.header('Other models')
df = df.sort_values(by=['F1'], ascending=False)
other_models = list(df['Model'].unique())
other_models.remove(df_best['Model'].iloc[0])

for each in other_models:

    st.subheader(each)
    df_other_model = df[(df['Model'] == each)]
    df_other_model_best_score = df_other_model[(df_other_model['F1'] == df_other_model['F1'].max())]
    df_other_model_best_score = df_other_model_best_score.iloc[0]

    col_best_1, col_best_2 = st.columns(2)
    with col_best_1: st.metric('F1', df_other_model_best_score['F1'], delta = df_other_model_best_score['F1']- df_best['F1'].iloc[0])
    with col_best_2: st.metric('F1_val', df_other_model_best_score['F1_val'])
    st.write("Parameters")
    st.dataframe(df_other_model_best_score)
    st.write("")