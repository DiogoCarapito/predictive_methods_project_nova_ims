# -*- coding: utf-8 -*-
"""3_Data Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lJIn6HWLVskjeE79VZ2wIYNvaX8OBW35
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

#!pip install streamlit
#!pip install streamlit_jupyter

# Bibliotecas inicias para setup
import sys
import streamlit as st

#verificar se estamos no colab ou não e importar coisas de acordo
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
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error

import inspect

# Page Setup
st.set_page_config(
    page_title="PMDM Grupo 11",
    page_icon=":sports_medal:",
    layout="wide")

st.title("Data Preprocessing")

radio_dataset = st.radio("Select Dataset:", ("Train", "Test"), horizontal=True)

if radio_dataset == "Train":
    full_path = path + "train.csv"
else:
    full_path = path + "test.csv"

# load dataset
df = pd.read_csv(full_path)

variaveis_nao_uteis = ['Outcome', 'RecordID', 'Athlete Id']
data_types = ['int64', 'float64']

variaveis_numericas = list(df.drop(variaveis_nao_uteis,axis=1).select_dtypes(include=data_types).columns)

variaveis_categoricas = list(df.select_dtypes(include='object').columns)

# Numeclatura de funções de preprocessamento

# errors_     >> tratamento de erros nos dados
# outlier_    >> tratamento de outliers
# missing_    >> tratamento de missing vaules
# engineer_   >> feature engeneering
# encoding_   >> encoding, one hot encoding
# scaling_    >> escalar variáveis
# select_     >> feature selection

# Dicionário de funções preprocesamento
preprocessing_functions = {}

# ERRORS

# substituir FASE e FALSE por False e TRUE por True (booleanos)

#função de processamento de errors no 'Mental preparation'
def errors_mental_preparation(df):

    # dicionário com antigo e novo parametro para substituit
    substitutes = {
        'FASE': False,
        'FALSE': False,
        'TRUE': True,
    }

    # substituição (obrigado ChatGPT)
    #df['Mental preparation'] = df['Mental preparation'].apply(lambda x: substitutes.get(x, x))
    df['Mental preparation'] = df['Mental preparation'].replace(substitutes)
    return df

# juntar ao dicionário 
preprocessing_functions['errors_mental_preparation'] = {
    'function': errors_mental_preparation,
    'description': 'substituir FASE e FALSE por False e TRUE por True',
    'type': 'error',
    'variables': ['Mental preparation'],
    'code': inspect.getsource(errors_mental_preparation),
}

# substituir '0' por '0-35' no Age group ('0-35' é o mais frequente - 69,4%)
# mas pode-se eliminar tmabém, pois corresponde a 0,13% dos dados

# processamento gravado em função
def errors_age_group(df):
    #dicionário com antigo e novo parametro para substituir
    substitute = {
        '0': '0-35',
    }

    # substituição
    #df['Age group'] = df['Age group'].apply(lambda x: substitutes.get(x, x))
    df['Age group'] = df['Age group'].replace(substitute)
    return df

# juntar ao dicionário de funções
preprocessing_functions['errors_age_group'] = {
    'function': errors_age_group,
    'description': "substituir '0' por '0-35' no Age group ('0-35' é o mais frequente - 69,4%)",
    'type': 'error',
    'variables':['Age group'],
    'code': inspect.getsource(errors_age_group),
}

# Athlete score -30.0 passa a 0

def error_athelete_score_negative(df):
    df['Athlete score'] = df['Athlete score'].apply(lambda x: 0 if x < 0 else x).astype(float)
    return df

# juntar ao dicionário de funções
preprocessing_functions['error_athelete_score_negative'] = {
    'function': error_athelete_score_negative,
    'description': "Athlete score -30.0 passa a 0",
    'type': 'error',
    'variables':['Athlete score'],
    'code': inspect.getsource(error_athelete_score_negative),
}

# Physiotherapy -50.0 passa a 0

def error_physiotherapy_negative(df):
    df['Physiotherapy'] = df['Physiotherapy'].apply(lambda x: 0 if x < 0 else x)
    return df

# juntar ao dicionário de funções
preprocessing_functions['error_physiotherapy_negative'] = {
    'function': error_physiotherapy_negative,
    'description': "Physiotherapy -50.0 passa a 0",
    'type': 'error',
    'variables':['Physiotherapy'],
    'code': inspect.getsource(error_physiotherapy_negative),
}

# Athlete score acima de 100 passa a 100 e abaixo de 0 fica 0

def error_athelete_score_over_100(df):
    df['Athlete score'] = df['Athlete score'].apply(lambda x: 100 if x > 100 else x).astype(float)
    return df

# juntar ao dicionário de funções
preprocessing_functions['error_athelete_score_over_100'] = {
    'function': error_athelete_score_over_100,
    'description': "Athlete score acima de 100 passa a 100 e abaixo de 0 fica 0",
    'type': 'error',
    'variables':['Athlete score'],
    'code': inspect.getsource(error_athelete_score_over_100),
}

# Ver erros do df_test e colocar nas funções para serem corrigidos
# apar3entemente não há necessidade...

# OUTLIERS

# ainda falta lool

# MISSING VALUES

## Número de eventos por variável categórica ('object')
#para visaulizar os missing values e que valores são mais frequentes para poder substituir

#for each in df.select_dtypes(include='object').columns:
#    st.write(each)
#    st.write(df[each].value_counts(dropna=False))
#    st.write('')

# Tratamento de missing values em variaveis categóricas pelo método de substituição por valor de maior frequência

def missing_categorical_substitute_highest_frequency(df):
    dict_variavel_sub = {
        'Age group':'0-35',
        'Disability': False,
        'Late enrollment': False,
        'Cancelled enrollment': False,
        'Mental preparation': False,
        'Outdoor Workout': False,
        'No coach': False,
    }
    # Substituição em cada variável o missing value pelo valor predefinido no dicionário
    for variavel, valor_sub in dict_variavel_sub.items():
        df[variavel] = df[variavel].fillna(valor_sub)
    return df

# juntar ao dicionário de funções
preprocessing_functions['missing_categorical_substitute_highest_frequency'] = {
    'function': missing_categorical_substitute_highest_frequency,
    'description': "Tratamento de missing values em variaveis categóricas pelo método de substituição por valor de maior frequência",
    'type': 'missing',
    'variables':[
        'Age group',
        'Disability',
        'Late enrollment',
        'Cancelled enrollment',
        'Mental preparation',
        'Outdoor Workout',
        'No coach',
    ],
    'code': inspect.getsource(missing_categorical_substitute_highest_frequency),
}

# Remover todos os missing values
# Estratégia temporária só para teatar modelos até encontrar melhor solução

def missing_drop_all(df):
    return df.dropna()

# juntar ao dicionário de funções
preprocessing_functions['missing_drop_all'] = {
    'function': missing_drop_all,
    'description': "Remover todos os missing values",
    'type': 'missing',
    'variables':[],
    'code': inspect.getsource(missing_drop_all),
}

# Numericos
# KNN

# Categoricos
# KNN

#FEATURE ENGINEERING

# eventualmente trasfomrmar em booleano?
# a logica é se existir a presença destes, então isso basta.
# tudo o que for superior a 0 passa a ser 1

def engineer_skewed_data(df):
    # lista de variáveis que vão ser processadas
    skewed_data = [
        'Previous attempts',
        'Sand training',
        'Plyometric training',
        'Other training',]

    # iteração pelo skewed data
    for each in skewed_data:
        # substituição de tudo o que for superior a 0 passar a 1
        df[each] = df[each].apply(lambda x: 1 if x > 0 else x)
    return df

# juntar ao dicionário de funções
preprocessing_functions['engineer_skewed_data'] = {
    'function': engineer_skewed_data,
    'description': "transformar varias variaveis muito skewed em 0 e 1",
    'type': 'engineer',
    'variables':[
        'Previous attempts',
        'Sand training',
        'Plyometric training',
        'Other training',
        ],
    'code': inspect.getsource(engineer_skewed_data),
}

# Criar uma variável chamada Total Training
# É uma soma de Cardiovascular training, Other training, Plyometric training, Sand training, Sport-specific training, Squad training, Strength training

def engineer_total_training(df):
    # lista de variáveis que vão ser somadas
    training = [
        'Cardiovascular training',
        'Other training',
        'Plyometric training',
        'Sand training',
        'Sport-specific training',
        'Squad training',
        'Strength training'
        ]

    df['Total training'] = df.apply(lambda x: sum(x[col] for col in training), axis=1)
    return df

# juntar ao dicionário de funções
preprocessing_functions['engineer_total_training'] = {
    'function': engineer_total_training,
    'description': "Criar uma variável chamada Total Training. É uma soma de Cardiovascular training, Other training, Plyometric training, Sand training, Sport-specific training, Squad training, Strength training",
    'type': 'engineer',
    'variables':[
        'Cardiovascular training',
        'Other training',
        'Plyometric training',
        'Sand training',
        'Sport-specific training',
        'Squad training',
        'Strength training'
        ],
    'code': inspect.getsource(engineer_total_training),
}

# Region simplification em continentes

def engineer_region_by_continent(df):
    dict_correspondencias_continentes = {
        'North America': 'America',
        'South America': 'America',
        'Central America': 'America',
        'Western Europe': 'Europe',
        'Eastern Europe': 'Europe',
        'Southern Europe': 'Europe',
        'East Asia': 'Asia',
        'Middle East': 'Asia',
        'Central Asia': 'Asia',
        'South Asia': 'Asia',
        'Northern Africa': 'Africa',
        'Southern Africa': 'Africa',
        'Oceania': 'Oceania',
    }
    df['Region'] = df['Region'].replace(dict_correspondencias_continentes)
    return df

# juntar ao dicionário de funções
preprocessing_functions['engineer_region_by_continent'] = {
    'function': engineer_region_by_continent,
    'description': "Region simplification em continentes",
    'type': 'engineer',
    'variables':['Region'],
    'code': inspect.getsource(engineer_region_by_continent),
}

# simplificação das competiçoe em nacional e internacional

def engineer_competition_national_international(df):
    # lista das novas correspondências
    dict_correspondencias_competicao = {
        'Local Match': 'National',
        'Regional Tournament': 'National',
        'Federation League': 'National',
        'National Cup': 'National',
        'Continental Championship': 'International',
        'World Championship': 'International',
        'Olympic Games': 'International', 
    }
    # substituição
    df['Competition'] = df['Competition'].replace(dict_correspondencias_competicao)
    return df

# juntar ao dicionário de funções
preprocessing_functions['engineer_competition_national_international'] = {
    'function': engineer_competition_national_international,
    'description': "simplificação das competiçoe em nacional e internacional",
    'type': 'engineer',
    'variables':['Competition'],
    'code': inspect.getsource(engineer_competition_national_international),
}

# simplificação das competiçoe em local, nacional e internacional

def engineer_competition_local_national_international(df):
    # lista das novas correspondências
    dict_correspondencias_competicao = {
        'Local Match': 'Local',
        'Regional Tournament': 'Local',
        'Federation League': 'National',
        'National Cup': 'National',
        'Continental Championship': 'International',
        'World Championship': 'International',
        'Olympic Games': 'International', 
    }
    # substituição
    df['Competition'] = df['Competition'].replace(dict_correspondencias_competicao)
    return df

# juntar ao dicionário de funções
preprocessing_functions['engineer_competition_local_national_international'] = {
    'function': engineer_competition_local_national_international,
    'description': "simplificação das competiçoe em local, nacional e internacional",
    'type': 'engineer',
    'variables':['Competition'],
    'code': inspect.getsource(engineer_competition_local_national_international),
}

#ENCODING

# transformar True e False em 1 e 0

def encoding_true_false_to_1_0(df):
    #  lista de variaveis a tratar
    list_true_false_to_1_0 = [
        'Past injuries',
        'Outdoor Workout',
        'Mental preparation',
        'Cancelled enrollment',
        'Late enrollment',
        'Disability',
    ]
    
    # iteração sobre variaveis
    for variable in list_true_false_to_1_0:
        
        #substituição de valores acima de 0 para 1
        df[variable] = df[variable].replace({True: 1, False: 0})

    return df

# juntar ao dicionário de funções
preprocessing_functions['encoding_true_false_to_1_0'] = {
    'function': encoding_true_false_to_1_0,
    'description': "transformar True e False em 1 e 0",
    'type': 'encoding',
    'variables':[
        'Past injuries',
        'Outdoor Workout',
        'Mental preparation',
        'Cancelled enrollment',
        'Late enrollment',
        'Disability',
    ],
    'code': inspect.getsource(encoding_true_false_to_1_0),
}

# One Hot Encoding categorical

def encoding_one_hot_encoding_categorical(df):
    #Lista de variáveis categoricas para one hot encoding
    list_one_hot_encoding = [
        'Competition',
        'Region',
        'Education',
        'Age group',
        'Income',
        'Sex',
        ]
    
    # filtrar do dataframe original só as variaveis para one hot
    df_one_hot_encoding = df[list_one_hot_encoding]
    
    # one hot encoding, eliminando a primeira coluna de forma a não dar erro
    df_one_hot_encoded = pd.get_dummies(df_one_hot_encoding, drop_first = True)
    
    # Junção das novas colunas com o dataframe original
    df_merged = df.merge(df_one_hot_encoded, left_index=True, right_index=True)

    # remoção das colunas antingas
    ## Está a dar erro...
    #df_merged = df_merged.drop(list_one_hot_encoding)

    return df_merged

# juntar ao dicionário de funções
preprocessing_functions['encoding_one_hot_encoding_categorical'] = {
    'function': encoding_one_hot_encoding_categorical,
    'description': "One Hot Encoding categorical",
    'type': 'encoding',
    'variables':[
        'Competition',
        'Region',
        'Education',
        'Age group',
        'Income',
        'Sex',
        ],
    'code': inspect.getsource(encoding_one_hot_encoding_categorical),
}

#SCALING

# Transformação de variáveis para o logaritmo para tratar skewness

def scaling_log_numerical(df):

    # Lista de variáveis
    log_transforms = [
        'Train bf competition',
        'Strength training',
        'Sand training',
        'Recovery',
        'Supplements',
        'Cardiovascular training',
        'Squad training',
        'Physiotherapy',
        'Plyometric training',
        'Sport-specific training',
        'Other training',
        'Total training']

    for variable in log_transforms:
        # aplicação do logaritmo
        try:
            df[variable] = np.log(df[variable]+0.01)
        except:
            pass

    return df

# juntar ao dicionário de funções
preprocessing_functions['scaling_log_numerical'] = {
    'function': scaling_log_numerical,
    'description': "Transformação de variáveis para o logaritmo para tratar skewness",
    'type': 'scaling',
    'variables':[
        'Train bf competition',
        'Strength training',
        'Sand training',
        'Recovery',
        'Supplements',
        'Cardiovascular training',
        'Squad training',
        'Physiotherapy',
        'Plyometric training',
        'Sport-specific training',
        'Other training',
        'Total training'
        ],
    'code': inspect.getsource(scaling_log_numerical),
}

# Transformação de variáveis para a raiz quadrada para tratar skewness
def scaling_sqrt_numerical(df):

    # Lista de variáveis
    sqrt_transforms = [
        'Train bf competition',
        'Strength training',
        'Sand training',
        'Recovery',
        'Supplements',
        'Cardiovascular training',
        'Squad training',
        'Physiotherapy',
        'Plyometric training',
        'Sport-specific training',
        'Other training',
        'Total training']

    for variable in sqrt_transforms:
        # aplicação do sqrt
        try:
            df[variable] = np.sqrt(df[variable])
        except:
            pass

    return df

# juntar ao dicionário de funções
preprocessing_functions['scaling_sqrt_numerical'] = {
    'function': scaling_sqrt_numerical,
    'description': "Transformação de variáveis para a raiz quadrada para tratar skewness",
    'type': 'scaling',
    'variables':[
        'Train bf competition',
        'Strength training',
        'Sand training',
        'Recovery',
        'Supplements',
        'Cardiovascular training',
        'Squad training',
        'Physiotherapy',
        'Plyometric training',
        'Sport-specific training',
        'Other training',
        'Total training'
        ],
    'code': inspect.getsource(scaling_sqrt_numerical),
}

# normalização com MinMax
from sklearn.preprocessing import MinMaxScaler
def scaling_numerical_min_max(df):
    
    # definir MinMax
    minmax_scaler = MinMaxScaler()
    
    # filtrar dataframe apenas pelas variáveis numéricas previamente definidas
    df_numericas = df[variaveis_numericas]
    
    # execução do MinMax
    minmax_scaler = minmax_scaler.fit(df_numericas)
    minmax_data = minmax_scaler.transform(df_numericas)
    
    # Transforma resultado num dataframe
    minmax_data = pd.DataFrame(minmax_data, columns = df_numericas.columns, index = df_numericas.index)

    # Substituir valores no dataframe original para novos valores MinMax
    df[variaveis_numericas] = minmax_data
    
    return df

# juntar ao dicionário de funções
preprocessing_functions['scaling_numerical_min_max'] = {
    'function': scaling_numerical_min_max,
    'description': "normalização com MinMax",
    'type': 'scaling',
    'variables':variaveis_numericas,
    'code': inspect.getsource(scaling_numerical_min_max),
}

#SELECTION

# deixar cair No Coach por só ter 2 valores positivos, vatiável quase uniforme
def selection_no_coach_drop(df):
    try:
        return df.drop('No coach')
    except:
        return df

# juntar ao dicionário de funções
preprocessing_functions['selection_no_coach_drop'] = {
    'function': selection_no_coach_drop,
    'description': "deixar cair No Coach por só ter 2 valores positivos, vatiável quase uniforme",
    'type': 'selection',
    'variables':'No coach',
    'code': inspect.getsource(selection_no_coach_drop),
}


# mostrar todas as funções existentes
# Pronto para copiar par uma lista, até tem a virgula e tudo xD
if 'preprocessing_func_list' not in st.session_state:
    st.session_state['preprocessing_func_list'] = {}

tab_function_description, tab_function_selection = st.tabs(['Descrição', 'Seleção'])

with tab_function_description:
    for key, value in preprocessing_functions.items():
        st.subheader(key)
        st.write('**Descrição**: ' + value['description'])
        st.code(value['code'], language = 'python')

with tab_function_selection:
    for key, value in preprocessing_functions.items():
        st.session_state['preprocessing_func_list'][key] = st.checkbox(key)

    preprocessing_function_list = []
    for key in st.session_state['preprocessing_func_list']:
        if st.session_state['preprocessing_func_list'][key] == True:
            preprocessing_function_list.append(preprocessing_functions[key]['function'])


    # Definição de um pipleine
    # deve ser passado um dataframe para ser processado
    # e uma lista de funções para serem executadas por ordem

    def pileline(df,function_list):
        # iteração pelas funçoes da lista
        for func in function_list:
            # execução da função
            df = func(df)
        return df

    # Aplicar pipeline ao df_train com a lista de funções 1
    df_preprocessed_1 = pileline(df,preprocessing_function_list)

    st.write('')
    st.dataframe(df_preprocessed_1.head())

    st.write(f"Total de dados: **{df.shape[0]}**")
    st.write(f"Total de dados sem missing values: **{df.dropna().shape[0]}**")
    st.write(f"Percentagem de linhas com missing values: **{round(100 * (1 - df.dropna().shape[0] / df.shape[0]), 2)}%**")

    if st.button('Guardar seleção de funções'):
        with open('funct_list.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the list to the CSV file
            writer.writerow(preprocessing_function_list)
        st.success('Guardado em funct_list.csv', icon="✅")