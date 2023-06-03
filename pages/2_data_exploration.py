# -*- coding: utf-8 -*-
"""2_Data Exploration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fxl_5EmeyiN4rPT8pRz26qnR-IB8JbNP
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

#!pip install streamlit
#!pip install streamlit_jupyter

# Bibliotecas inicias para setup
import sys
import streamlit as st

# verificar se estamos no colab ou não e importar coisas de acordo
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

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# Page Setup
st.set_page_config(
    page_title="PMDM Grupo 11",
    page_icon=":sports_medal:",
    layout="wide")

st.title("Data Exploration")

# selecionar entre Train e Test dataset 
radio_dataset = st.radio("Select Dataset:", ("Train", "Test"), horizontal=True)

if radio_dataset == "Train":
    full_path = path + "train.csv"
else:
    full_path = path + "test.csv"

# load dataset
df = pd.read_csv(full_path)

# show head
st.header(".head()")
st.table(df.head())

st.write(f"Número de linhas: **{df.shape[0]}**")
st.write(f"Número de variáveis: **{df.shape[1]}**")

# Descrição de cada variável
description = {
"Athlete Id": "ID",
"Age group": "Athlete age range",
"Athlete score": "Athlete score from previous competitions",
"Cancelled enrollment": "Athlete cancelled the competition enrollment",
"Cardiovascular training": "Number of training sessions such as running, cycling, or swimming",
"Competition": "Type of competition",
"Disability": "Athlete with disability",
"Edition": "The year of the edition competition",
"Education": "Athlete education level",
"Income": "Athlete income level",
"Late enrollment": "Athlete enrolled in the competition belatedly",
"Mental preparation": "Athlete has developed strategies for handling with stress and pressure",
"No coach": "Athlete does not have a coach",
"Other training": "Number of training sessions using non-standard approaches",
"Outcome": "Competition result (TARGET)",
"Outdoor Workout": "Training conducted outdoors in parks or forests",
"Past injuries": "Athlete had sport injuries",
"Physiotherapy": "Number of physiotherapy sessions",
"Plyometric training": "Number of training sessions involving explosive, high-intensity movements",
"Previous attempts": "Number of previous competitions attempts",
"RecordID": "ID of the registration of one athlete into an edition of a given competition",
"Recovery": "Number of recovery sessions using stretching and massages techniques",
"Region": "Athlete region",
"Sand training": "Number of training sessions  involving sand drills",
"Sex": "Athlete sex",
"Sport-specific training": "Number of training sessions that mimic competition scenarios",
"Squad training": "Number of training sessions that involve a group of athletes working together to prepare for competition",
"Strength training": "Number of training sessions using weightlifting and bodyweight exercises",
"Supplements": "Number of nutritional supplements taken to aid performance",
"Train bf competition": "Number of pre-competition preparation sessions",
}

# Descrição de cada variável
st.header("Descrição de cada variável")
#st.table(pd.DataFrame.from_dict(description, orient='index', columns=['Description']))
st.table(pd.concat([
    pd.DataFrame.from_dict(description, orient='index', columns=['Descrição']),
    df.dtypes.rename("Tipo de dados").replace({"object": "Categórico", "int64": "Numérico", "float64": "Numérico", "bool": "Booleano"}),
    ], axis=1))

# desceibe numerical variables
st.header(".describe().T")
st.table(df.describe().T)

# describe categorical variables
st.header(".describe(include='object').T")
st.table(df.describe(include='object').T)

# Missing values
st.header("Contagem e percentagem de missing values")
st.table(pd.concat([
    df.isna().sum(),
    round(df.isna().sum()/len(df)*100, 3).apply(lambda x: '{:.3f}%'.format(x)),
    ], axis=1)
)

st.write(f"Total de dados: **{df.shape[0]}**")
st.write(f"Total de dados sem missing values: **{df.dropna().shape[0]}**")
st.write(f"Percentagem de linhas com missing values: **{round(100*(1-df.dropna().shape[0]/df.shape[0]),2)}%**")

# dados duplicados
st.header("Deteção de duplicados")
st.write(f"Número de duplicados: **{df.duplicated().sum()}**")
st.write(f"Número de duplicados no RecordID: **{df.RecordID.duplicated().sum()}**")
st.write(f"Número de duplicados excluindo RecordID: **{df.drop(['RecordID'], axis=1).duplicated().sum()}**")
st.write(f"Número de duplicados excluindo e Athlete Id: **{df.drop(['RecordID', 'Athlete Id'], axis=1).duplicated().sum()}**")
st.write("**Não parece haver duplicados!**")

variaveis_numericas = list(df.select_dtypes(include=['int64','float64']).columns)
st.header('Variáveis numéricas')
st.table(variaveis_numericas)

def plot_multiple_histograms(data, feats, title="Numeric Variables' Histograms"):

    # Prepare figure. Create individual axes where each histogram will be placed
    fig, axes = plt.subplots(2, np.ceil(len(feats) / 2).astype(int), figsize=(20, 11))

    # Plot data
    # Iterate across axes objects and associate each histogram
    for ax, feat in zip(axes.flatten(), feats):
        ax.hist(data[feat])
        ax.set_title(feat)

    # Layout
    # Add a centered title to the figure
    fig.suptitle(title)

    # Display the plot in Streamlit using st.pyplot()
    st.pyplot(fig)

    return


## Define a function that plots multiple box plots

def plot_multiple_boxplots(data, feats, title="Numeric Variables' Box Plots"):

    # Prepare figure. Create individual axes where each boxplot will be placed
    fig, axes = plt.subplots(2, np.ceil(len(feats) / 2).astype(int), figsize=(20, 11))

    # Plot data
    # Iterate across axes objects and associate each boxplot
    for ax, feat in zip(axes.flatten(), feats):
        sns.boxplot(x=data[feat], ax=ax)
        ax.set_title(feat)

    # Layout
    # Add a centered title to the figure
    fig.suptitle(title)

    # Display the plot in Streamlit using st.pyplot()
    st.pyplot(fig)

    return

sns.set()

plot_multiple_histograms(df, variaveis_numericas)
plot_multiple_boxplots(df, variaveis_numericas)

variaveis_categoricas = list(df.select_dtypes(include='object').columns)
st.header('Variáveis categóricas')
st.table(variaveis_categoricas)

variaveis_booleanas = list(df.select_dtypes(include='bool').columns)
st.header('Variáveis booleanas')
st.table(variaveis_booleanas)


#pair plot
st.header("Pairwise Relationship of Numerical Variables")

if st.checkbox('Executar Pairplot',value=False,) == True:
    sns.set()
    pairplot = sns.pairplot(df[variaveis_numericas], diag_kind="hist")
    pairplot.fig.subplots_adjust(top=0.95)
    pairplot.fig.suptitle("Pairwise Relationship of Numerical Variables", fontsize=20)

    # Display the plot in Streamlit
    st.pyplot(pairplot.fig)
else:
    st.write("Clicar para executar pairplot")

