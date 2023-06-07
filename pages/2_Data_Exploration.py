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
from scipy.stats import skew

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
st.write("Visualização tabular das primeiras linhas do dataset")
st.table(df.head())

#st.write(f"Número de linhas: **{df.shape[0]}**")
#st.write(f"Número de variáveis: **{df.shape[1]}**")

col_metrics_1, col_metrics_2 = st.columns(2)
with col_metrics_1:
    st.metric("Número de linhas", df.shape[0])
with col_metrics_2:
    st.metric("Número de variáveis", df.shape[1])

st.write("----")

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
st.header("Lista de variáveis")
st.write("Descrição de cada variável")
#st.table(pd.DataFrame.from_dict(description, orient='index', columns=['Description']))
st.table(pd.concat([
    pd.DataFrame.from_dict(description, orient='index', columns=['Descrição']),
    df.dtypes.rename("Tipo de dados").replace({"object": "Categórico", "int64": "Numérico", "float64": "Numérico", "bool": "Booleano"}),
    ], axis=1))

st.write("Notas:")
st.write("- Só existem variáveis numéricas e categóricas, aparentemente não há variaveis booleanas, mas possível que haja variáveis categóricas com apenas 2 valores possíveis ou variáveis numéricas com apenas 0 e 1.")
st.write("- Athelte Id e RecordID são variáveis de identificação, logo não são relevantes para o modelo.")
st.write("- Outcome é a variável target, que é numérica mas varia entre 0 e 1. Portanto, é um problema de Classificação")
st.write("----")



# Missing values
st.header("Missing values")
st.table(pd.concat([
    df.isna().sum(),
    round(df.isna().sum()/len(df)*100, 3).apply(lambda x: '{:.3f}%'.format(x)),
    ], axis=1)
)

#st.write(f"Total de dados: **{df.shape[0]}**")
#st.write(f"Total de dados sem missing values: **{df.dropna().shape[0]}**")
#st.write(f"Percentagem de linhas com missing values: **{round(100*(1-df.dropna().shape[0]/df.shape[0]),2)}%**")

col_mv_1, col_mv_2, col_mv_3 = st.columns(3)
with col_mv_1:
    st.metric("Total de dados:",df.shape[0])
with col_mv_2:
    st.metric("Total de dados sem missing values:",df.dropna().shape[0])
with col_mv_3:
    st.metric("Percentagem de linhas com missing values:",str(round(100*(1-df.dropna().shape[0]/df.shape[0]),2))+"%")
#st.write(f"Total de dados: **{df.shape[0]}**")
#st.write(f"Total de dados sem missing values: **{df.dropna().shape[0]}**")
#st.write(f"Percentagem de linhas com missing values: **{round(100*(1-df.dropna().shape[0]/df.shape[0]),2)}%**")


st.write("Notas:")
st.write("- Todas as variáveis têm missing values, excepto Outcome e RecordID.")
st.write("- Se removessemos todas as linhas com missing values, perdiamos 13,% dos dados.")
st.write("- Será importante tratar os missing.")
st.write("----")



# dados duplicados
st.header("Deteção de duplicados")
col_dup_1, col_dup_2, col_dup_3, col_dup_4 = st.columns(4)
with col_dup_1:
    st.metric("Nº duplicados",df.duplicated().sum())
with col_dup_2:
    st.metric("Nº duplicados no RecordID",df.RecordID.duplicated().sum())
with col_dup_3:
    st.metric("Nº dup. excluindo RecordID",df.drop(['RecordID'], axis=1).duplicated().sum())
with col_dup_4:
    st.metric("Nº dup. ex. RecordID e Athlete Id",df.drop(['RecordID', 'Athlete Id'], axis=1).duplicated().sum())
#st.write(f"Número de duplicados: **{df.duplicated().sum()}**")
#st.write(f"Número de duplicados no RecordID: **{df.RecordID.duplicated().sum()}**")
#st.write(f"Número de duplicados excluindo RecordID: **{df.drop(['RecordID'], axis=1).duplicated().sum()}**")
#st.write(f"Número de duplicados excluindo e Athlete Id: **{df.drop(['RecordID', 'Athlete Id'], axis=1).duplicated().sum()}**")
st.write("**Não parece haver duplicados!**")
st.write("----")

# transformar variáveis booleanas que estão originalmente como object
for each in ['Disability', 'Late enrollment', 'Cancelled enrollment', 'Outdoor Workout', 'No coach', 'Past injuries']:
    df[each] = df[each].astype(bool)

# desceibe numerical variables
st.header(".describe().T")
st.write("Visão descritiva básica de variáveis numéricas")
st.table(df.describe().T)
st.write("Notas:")
st.write("- há variáveis têm um standard deviation e valores máximso muito elevado, o que indica que os valores estão muito dispersos e/ou existência de outliers.")
st.write("- há variáveis com valores mínimos abaixo de 0 (Athlete score e Physiotherapy), o que não faz sentido e vai ser necessário corrigir.")
st.write("----")

# describe categorical variables
st.header(".describe(include='object').T")
st.write("Visão descritiva básica de variáveis categoricas")
st.table(df.describe(include='object').T)
st.write("Notas:")
st.write("- Afinal já existem variáveis booleanas, mal interpretadas inicialmente por categóticas.")
st.write("- Mental preparation, No coach, Outdoor Workout, Past injuries, Cancelled enrollment, Late enrollment e Disability são variáveis booleanas.")
st.write("- Existem variáveis com muitos valores únicos, poderá ser necessário reduzir.")
st.write("- Sex poderá ser considerada uma variável booleana se for necessário processar dessa forma dependendo do modelo.")
st.write("----")


variaveis_numericas = list(df.select_dtypes(include=['int64','float64']).columns)
st.header('Variáveis numéricas')
st.write(variaveis_numericas)

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

st.subheader("Histogramas")
plot_multiple_histograms(df, variaveis_numericas)
st.subheader("Boxplots")
plot_multiple_boxplots(df, variaveis_numericas)


st.subheader("Skewness")
# Calculate skewness
skewness = df.skew()
skewness_sorted = skewness.sort_values(ascending=False)
skewness_sorted = skewness_sorted.drop('No coach')

fig = plt.figure(figsize=(10, 6))
skewness_sorted.plot(kind='bar')
plt.xlabel('Variable')
plt.ylabel('Skewness')
plt.title('Skewness of Variables')
plt.xticks(rotation=90)
st.pyplot(fig)


skewed_vars = skewness[skewness > 5].index.tolist()
skewed_vars.remove('No coach')
skewed_vars.remove('Late enrollment')
df_train_1 = df[df['Outcome'] == 1]
df_train_0 = df[df['Outcome'] == 0]


# Calculate the number of rows and columns for the grid
num_vars = len(skewed_vars)
num_cols = min(1, num_vars)  # Set the maximum number of columns to 2
num_rows = int(np.ceil(num_vars / num_cols))

# Create the subplots grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 40))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the skewed variables
for i, var in enumerate(skewed_vars):
    ax = axes[i]

    # Plot the scatter plots
    sns.scatterplot(x=range(len(df_train_1[var])), y=df_train_1[var], label='Outcome 1', ax=ax)
    sns.scatterplot(x=range(len(df_train_0[var])), y=df_train_0[var], label='Outcome 0', ax=ax)

    # Set the title and labels
    ax.set_title("Scatter Plot of " + var, fontsize=16)
    ax.set_xlabel("Index", fontsize=14)
    ax.set_ylabel(var, fontsize=14)

# Remove any unused subplots
if num_vars < len(axes):
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

# Adjust spacing between subplots
fig.tight_layout()

# Display the grid of scatter plots
st.pyplot(fig)

st.write("----")

variaveis_categoricas = list(df.select_dtypes(include='object').columns)
st.header('Variáveis categóricas')
st.write(variaveis_categoricas)

def plot_multiple_barplots(data, feats, title):
    fig, axes = plt.subplots(2, np.ceil(len(feats) / 2).astype(int), figsize=(20, 11))

    for ax, feat in zip(axes.flatten(), feats):
        sns.countplot(x=data[feat], ax=ax)
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle(title)

    st.pyplot(fig)

    return

try:
    plot_multiple_barplots(df, variaveis_categoricas, "Categorical Variables' Bar Plots")
except:
    st.write("Erro ao executar plot_multiple_barplots")


st.write("----")
variaveis_booleanas = list(df.select_dtypes(include='bool').columns)
st.header('Variáveis booleanas')
st.write(variaveis_booleanas)
plot_multiple_barplots(df, variaveis_booleanas, "Boolean Variables' Bar Plots")


st.write("----")
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

