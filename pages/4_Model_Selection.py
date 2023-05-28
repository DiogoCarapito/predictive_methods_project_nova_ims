# First imports
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
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import csv
import inspect

# sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, \
    mean_absolute_error, mean_squared_error

#from itertools import product


# Page Setup
st.title("Model Selection")

# Tabs
tab_logistic, tab_decision_tree, tab_ensambles, tab_neural_network =  st.tabs(['LogisticRegression', 'DecisionTree', 'Ensembles', 'NeuralNetwork'])

# Data Loading
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')

# Lista de variáveis
target = 'Outcome'

variaveis_nao_uteis = [
    'RecordID',
    'Athlete Id'
]

variaveis_numericas = list(df_train.drop(variaveis_nao_uteis, axis=1).select_dtypes(include=['int', 'float']).columns)

variaveis_booleanas = [
    'Disability',
    'Late enrollment',
    'Cancelled enrollment',
    'Mental preparation',
    'Outdoor Workout',
    'No coach',
    'Past injuries'
]

variaveis_categoricas = [
    'Competition',
    'Sex',
    'Region',
    'Education',
    'Age group',
    'Income'
]

# Numeclatura de funções de preprocessamento

# error_     >> tratamento de erros nos dados
# outlier_    >> tratamento de outliers
# missing_    >> tratamento de missing vaules
# engineer_   >> feature engeneering
# encoding_   >> encoding, one hot encoding
# scaling_    >> escalar variáveis
# select_     >> feature selection

# Dicionário de funções preprocesamento
preprocessing_functions = {}


# ERRORS

# remover varaiveis não úteis no modelo preditivo
def remove_ids(df):
    variaveis_nao_uteis = ['RecordID', 'Athlete Id']
    df = df.drop(variaveis_nao_uteis, axis=1)
    return df
# juntar ao dicionário
preprocessing_functions['remove_ids'] = {
    'function': remove_ids,
    'description': 'remover varaiveis não úteis no modelo preditivo',
    'type': 'remove',
    'variables': ['RecordID', 'Athlete Id'],
    'code': inspect.getsource(remove_ids),
}


# substituir FASE e FALSE por False e TRUE por True (booleanos)
# função de processamento de errors no 'Mental preparation'
def error_mental_preparation(df):
    # dicionário com antigo e novo parametro para substituit
    substitutes = {
        'FASE': False,
        'FALSE': False,
        'TRUE': True,
    }
    # df['Mental preparation'] = df['Mental preparation'].apply(lambda x: substitutes.get(x, x))
    df['Mental preparation'] = df['Mental preparation'].replace(substitutes)
    return df
# juntar ao dicionário
preprocessing_functions['error_mental_preparation'] = {
    'function': error_mental_preparation,
    'description': 'substituir FASE e FALSE por False e TRUE por True',
    'type': 'error',
    'variables': ['Mental preparation'],
    'code': inspect.getsource(error_mental_preparation),
}


# substituir FASE e FALSE por False e TRUE por True (booleanos)
# função de processamento de errors no 'Mental preparation'
def error_false_true(df):
    # dicionário com antigo e novo parametro para substituir
    substitutes = {
        'FASE': False,
        'FALSE': False,
        'TRUE': True,
    }
    # df['Mental preparation'] = df['Mental preparation'].apply(lambda x: substitutes.get(x, x))
    df = df.replace(substitutes)
    return df
# juntar ao dicionário
preprocessing_functions['error_false_true'] = {
    'function': error_false_true,
    'description': 'substituir FASE e FALSE por False e TRUE por True',
    'type': 'error',
    'variables': [],
    'code': inspect.getsource(error_false_true),
}


# substituir '0' por '0-35' no Age group ('0-35' é o mais frequente - 69,4%)
# mas pode-se eliminar tmabém, pois corresponde a 0,13% dos dados
# processamento gravado em função
def error_age_group(df):
    # dicionário com antigo e novo parametro para substituir
    substitute = {
        '0': '0-35',
    }

    # substituição
    # df['Age group'] = df['Age group'].apply(lambda x: substitutes.get(x, x))
    df['Age group'] = df['Age group'].replace(substitute)
    return df
# juntar ao dicionário de funções
preprocessing_functions['error_age_group'] = {
    'function': error_age_group,
    'description': "substituir '0' por '0-35' no Age group ('0-35' é o mais frequente - 69,4%)",
    'type': 'error',
    'variables': ['Age group'],
    'code': inspect.getsource(error_age_group),
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
    'variables': ['Athlete score'],
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
    'variables': ['Physiotherapy'],
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
    'variables': ['Athlete score'],
    'code': inspect.getsource(error_athelete_score_over_100),
}


# Late enrolement na realidade é uma Booleana desfarçada de 0 e 1
def error_late_enrolement_0_1(df):
    substitutes = {
        0: False,
        1: True,
    }
    df['Late enrollment'] = df['Late enrollment'].replace(substitutes)
    # actualizar lista de variaveis
    # variaveis_numericas = variaveis_numericas.drop('Late enrollment')
    # variaveis_booleanas.append('Late enrollment')
    return df
# juntar ao dicionário de funções
preprocessing_functions['error_late_enrolement_0_1'] = {
    'function': error_late_enrolement_0_1,
    'description': "Late enrolement na realidade é uma Booleana desfarçada de 0 e 1",
    'type': 'error',
    'variables': ['Late enrollment'],
    'code': inspect.getsource(error_late_enrolement_0_1),
}


# No coach na realidade é uma Booleana desfarçada de 0 e 1
def error_no_coach_0_1(df):
    substitutes = {
        0: False,
        1: True,
    }
    df['No coach'] = df['No coach'].replace(substitutes)
    # actualizar lista de variaveis
    # variaveis_numericas = variaveis_numericas.drop('No coach')
    # variaveis_booleanas.append('No coach')
    return df
# juntar ao dicionário de funções
preprocessing_functions['error_no_coach_0_1'] = {
    'function': error_no_coach_0_1,
    'description': "No coach na realidade é uma Booleana desfarçada de 0 e 1",
    'type': 'error',
    'variables': ['No coach'],
    'code': inspect.getsource(error_no_coach_0_1),
}


# Ver erros do df_test e colocar nas funções para serem corrigidos
# aparentemente não há necessidade...

# OUTLIERS

# fazer um clip nas variaveis com alto skewness, numero do clip escolhido visualmente
# numero foi escolhido conservador, pode ser que tenha que ser mais baixo...

def outlier_high_skewness(df):
    high_skewness_variables = {
        'Sand training': 500,
        'Recovery': 4000,
        'Cardiovascular training': 4000,
        'Squad training': 150,
        'Physiotherapy': 800,
        'Sport-specific training': 320,
        'Other training': 130,
    }
    # iterar
    for key, value in high_skewness_variables.items():
        df[key] = df[key].apply(lambda x: value if x > value else x).astype(float)
    return df
# juntar ao dicionário de funções
preprocessing_functions['outlier_high_skewness'] = {
    'function': outlier_high_skewness,
    'description': "fazer um clip nas variaveis com alto skewness, numero do clip escolhido visualmente",
    'type': 'error',
    'variables': [
        'Sand training',
        'Recovery',
        'Cardiovascular training',
        'Squad training',
        'Physiotherapy',
        'Sport-specific training',
        'Other training',
    ],
    'code': inspect.getsource(outlier_high_skewness),
}


# MISSING VALUES

## Número de eventos por variável categórica ('object')
# para visaulizar os missing values e que valores são mais frequentes para poder substituir

# for each in df.select_dtypes(include='object').columns:
#    st.write(each)
#    st.write(df[each].value_counts(dropna=False))
#    st.write('')

# Tratamento de missing values em variaveis categóricas pelo método de substituição por valor de maior frequência

def missing_categorical_substitute_highest_frequency(df):
    dict_variavel_sub = {
        'Age group': '0-35',
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
    'variables': [
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


# tratamento de missing balues nas variaveis numericas com knn inputer 5
def missing_numerical_knn_5(df):
    list_miising_numerical = [
        'Cardiovascular training',
        'Other training',
        'Train bf competition',
        'Supplements',
        'Plyometric training',
        'Edition', 'Sand training',
        'Physiotherapy',
        'Sport-specific training',
        'Previous attempts',
        'Recovery',
        'Strength training',
        'Squad training',
        'Athlete score',
    ]
    imputer = KNNImputer(n_neighbors=5)
    df[list_miising_numerical] = imputer.fit_transform(df[list_miising_numerical])
    return df
# juntar ao dicionário de funções
preprocessing_functions['missing_numerical_knn_5'] = {
    'function': missing_numerical_knn_5,
    'description': "tratamento de missing balues nas variaveis numericas com knn inputer",
    'type': 'missing',
    'variables': 'numericas',
    'code': inspect.getsource(missing_numerical_knn_5),
}


# tratamento de missing balues nas variaveis numericas com knn inputer 7
def missing_numerical_knn_7(df):
    list_miising_numerical = [
        'Cardiovascular training',
        'Other training',
        'Train bf competition',
        'Supplements',
        'Plyometric training',
        'Edition', 'Sand training',
        'Physiotherapy',
        'Sport-specific training',
        'Previous attempts',
        'Recovery',
        'Strength training',
        'Squad training',
        'Athlete score',
    ]

    imputer = KNNImputer(n_neighbors=7)
    df[list_miising_numerical] = imputer.fit_transform(df[list_miising_numerical])
    return df
# juntar ao dicionário de funções
preprocessing_functions['missing_numerical_knn_7'] = {
    'function': missing_numerical_knn_7,
    'description': "tratamento de missing balues nas variaveis numericas com knn inputer",
    'type': 'missing',
    'variables': 'numericas',
    'code': inspect.getsource(missing_numerical_knn_7),
}


# Categoricos
# KNN

# Remover todos os missing values
# Estratégia temporária só para teatar modelos até encontrar melhor solução

def missing_drop_all(df):
    return df.dropna()
# juntar ao dicionário de funções
preprocessing_functions['missing_drop_all'] = {
    'function': missing_drop_all,
    'description': "Remover todos os missing values",
    'type': 'missing',
    'variables': [],
    'code': inspect.getsource(missing_drop_all),
}


# FEATURE ENGINEERING

# eventualmente trasfomrmar em booleano?
# a logica é se existir a presença destes, então isso basta.
# tudo o que for superior a 0 passa a ser 1

def engineer_skewed_data(df):
    # lista de variáveis que vão ser processadas
    skewed_data = [
        'Previous attempts',
        'Sand training',
        'Plyometric training',
        'Other training', ]
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
    'variables': [
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
    'variables': [
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
    'variables': ['Region'],
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
    'variables': ['Competition'],
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
    'variables': ['Competition'],
    'code': inspect.getsource(engineer_competition_local_national_international),
}


# Passar physiotherapy para booleano
def engineer_physiotherapy_to_boolean(df):
    to_boolean_list = [
        'Physiotherapy'
    ]
    for column in to_boolean_list:
        df[column] = df[column].apply(lambda x: True if x > 0 else False).astype('bool')
    return df
preprocessing_functions['engineer_physiotherapy_to_boolean'] = {
    'function': engineer_physiotherapy_to_boolean,
    'description': "Passar physiotherapy para booleano",
    'type': 'engineer',
    'variables': ['Physiotherapy'],
    'code': inspect.getsource(engineer_physiotherapy_to_boolean),
}


# ENCODING

# transformar True e False em 1 e 0
def encoding_true_false_to_1_0(df):
    #  lista de variaveis a tratar
    list_true_false_to_1_0 = list(df_train.select_dtypes(include='bool').columns)

    # iteração sobre variaveis
    for variable in list_true_false_to_1_0:
        # substituição de valores acima de 0 para 1
        df[variable] = df[variable].replace({True: 1, False: 0})

    return df
# juntar ao dicionário de funções
preprocessing_functions['encoding_true_false_to_1_0'] = {
    'function': encoding_true_false_to_1_0,
    'description': "transformar True e False em 1 e 0",
    'type': 'encoding',
    'variables': [
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
    # Lista de variáveis categoricas para one hot encoding
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
    df_one_hot_encoded = pd.get_dummies(df_one_hot_encoding, drop_first=True)
    # Junção das novas colunas com o dataframe original
    df_merged = df.merge(df_one_hot_encoded, left_index=True, right_index=True)
    # remoção das colunas antigas
    df_merged = df_merged.drop(list_one_hot_encoding, axis=1)
    return df_merged
# juntar ao dicionário de funções
preprocessing_functions['encoding_one_hot_encoding_categorical'] = {
    'function': encoding_one_hot_encoding_categorical,
    'description': "One Hot Encoding categorical",
    'type': 'encoding',
    'variables': [
        'Competition',
        'Region',
        'Education',
        'Age group',
        'Income',
        'Sex',
    ],
    'code': inspect.getsource(encoding_one_hot_encoding_categorical),
}


# SCALING
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
            df[variable] = np.log(df[variable] + 0.01)
        except:
            pass
    return df
# juntar ao dicionário de funções
preprocessing_functions['scaling_log_numerical'] = {
    'function': scaling_log_numerical,
    'description': "Transformação de variáveis para o logaritmo para tratar skewness",
    'type': 'scaling',
    'variables': [
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
        'Total training'
    ]
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
    'variables': [
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
def scaling_numerical_min_max(df):
    # definir MinMax
    minmax_scaler = MinMaxScaler()
    # filtrar dataframe apenas pelas variáveis numéricas previamente definidas
    df_numericas = df[variaveis_numericas]
    # execução do MinMax
    minmax_scaler = minmax_scaler.fit(df_numericas)
    minmax_data = minmax_scaler.transform(df_numericas)
    # Transforma resultado num dataframe
    minmax_data = pd.DataFrame(minmax_data, columns=df_numericas.columns, index=df_numericas.index)
    # Substituir valores no dataframe original para novos valores MinMax
    df[variaveis_numericas] = minmax_data
    return df
# juntar ao dicionário de funções
preprocessing_functions['scaling_numerical_min_max'] = {
    'function': scaling_numerical_min_max,
    'description': "normalização com MinMax",
    'type': 'scaling',
    'variables': variaveis_numericas,
    'code': inspect.getsource(scaling_numerical_min_max),
}


# SELECTION

# deixar cair No Coach por só ter 2 valores positivos, vatiável quase uniforme
def selection_no_coach_drop(df):
    try:
        return df.drop('No coach', axis=1)
    except:
        return df
# juntar ao dicionário de funções
preprocessing_functions['selection_no_coach_drop'] = {
    'function': selection_no_coach_drop,
    'description': "deixar cair No Coach por só ter 2 valores positivos, vatiável quase uniforme",
    'type': 'selection',
    'variables': 'No coach',
    'code': inspect.getsource(selection_no_coach_drop),
}


# remover variaveis categóricas
def selection_categtical_drop(df):
    return df.select_dtypes(exclude='object')
# juntar ao dicionário de funções
preprocessing_functions['selection_no_coach_drop'] = {
    'function': selection_no_coach_drop,
    'description': "deixar cair No Coach por só ter 2 valores positivos, vatiável quase uniforme",
    'type': 'selection',
    'variables': 'No coach',
    'code': inspect.getsource(selection_no_coach_drop),
}



# PIPELINE
# Definição de um pipleine
# deve ser passado um dataframe para ser processado
# e uma lista de funções para serem executadas por ordem

def pileline(df, function_list):
    # iteração pelas funçoes da lista
    for func in function_list:
        # execução da função
        df = func(df)
    return df



# Tabs para cada modelo
with tab_logistic:
    st.header("Logistic Regression")
    st.write(preprocessing_functions)

with tab_decision_tree:
    st.header("Decision Tree")

with tab_ensambles:
    st.header("Ensembles Methods")

with tab_neural_network:
    st.header("Neural Networks")



#st.write(df_train.describe(include='object'))

# Mostrar exemplos no fim de cada função de preprocessamento
# mudar para False para ser mais rápida a execução do notebook
#test_mode = True

# criação de listas de tipos de variáveis
# Variáveis categóricas

# variaveis_categoricas = list(df_train.select_dtypes(include='object').columns)
# Variáveis não uteis para modelos



# variaveis_categoricas = list(df_train.drop(variaveis_nao_uteis,axis=1).select_dtypes(include="object").columns)

if test_mode is True:
    print('variaveis_numericas =', variaveis_numericas)
    print('variaveis_booleanas =', variaveis_booleanas)
    print('variaveis_categoricas =', variaveis_categoricas)
    print('variaveis_nao_uteis', variaveis_nao_uteis)
    print('target =', target)




# mostrar todas as funções existentes
# Pronto para copiar par uma lista, até tem a virgula e tudo xD
if test_mode is True:
    for each in list(preprocessing_functions.keys()): print(each + ',')

# Pileine 1

# definir quais funções entram no pipeline
func_list = [
    remove_ids,
    error_mental_preparation,
    error_false_true,
    error_age_group,
    error_athelete_score_negative,
    error_physiotherapy_negative,
    error_athelete_score_over_100,
    error_late_enrolement_0_1,
    error_no_coach_0_1,
    outlier_high_skewness,
    missing_categorical_substitute_highest_frequency,
    missing_numerical_knn_5,
    # missing_numerical_knn_7,
    missing_drop_all,
    engineer_skewed_data,
    engineer_total_training,
    engineer_region_by_continent,
    # engineer_competition_national_international,
    engineer_competition_local_national_international,
    # engineer_physiotherapy_to_boolean,
    # encoding_true_false_to_1_0,
    encoding_one_hot_encoding_categorical,
    scaling_log_numerical,
    # scaling_sqrt_numerical,
    scaling_numerical_min_max,
    selection_no_coach_drop,
]

# Aplicar pipeline ao df_train com a lista de funções 1
df_train_preprocessed = pileline(df_train, func_list)

# visualizar resultado do preprocessamento implementado
if test_mode is True:
    df_train_preprocessed.head()

# df_train_preprocessed_1.isna().sum()
print(
    f'Número de missing values no depois do pós-processamento: {df_train_preprocessed.shape[0] - df_train_preprocessed.dropna().shape[0]}')
print(
    f'Total de missing values não tratado que foram removidos removidos: {df_train.shape[0] - df_train_preprocessed.shape[0]}')
# print(f"Percentagem de missing values pos-processamento: {round(100*(1-df_train_preprocessed.dropna().shape[0]/df_train_preprocessed.shape[0]),2)}%")
# print(f'Total de linhas original: {df_train.shape[0]}')
print(f'Total de linhas pós-processamento: {df_train_preprocessed.shape[0]}')
print(
    f'Percentagem de dados removidos durante pós.processamento: {round(100 * (df_train.shape[0] - df_train_preprocessed.shape[0]) / df_train.shape[0], 2)}%')

df_train_preprocessed.head()

df_train_preprocessed.describe()

# df_train_preprocessed.select_dtypes(exclude='object')
# df_train_preprocessed.select_dtypes(exclude=['int','float'])
# df_train_preprocessed.select_dtypes(include='bool')
df_train_preprocessed['Physiotherapy']

"""____

# Model Evaluation
"""

X = df_train_preprocessed.loc[:, df_train_preprocessed.columns != 'Outcome']
y = df_train_preprocessed['Outcome']

X = X.drop(list(df_train_preprocessed.select_dtypes(include='object').columns), axis=1)

if test_mode is True:
    X.describe()


def run_model_LR(X, y):
    model = LogisticRegression().fit(X, y)
    return model


def evaluate_model(X, y, model):
    return model.score(X, y)


kf = KFold(n_splits=5)


def avg_score_LR(method, X, y):
    score_train = []
    score_test = []

    # method: KFold
    for train_index, test_index in method.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # run_model_LR function
        model = run_model_LR(X_train, y_train)

        # evaluate_model function
        value_train = evaluate_model(X_train, y_train, model)
        value_test = evaluate_model(X_test, y_test, model)

        score_train.append(value_train)
        score_test.append(value_test)
    if test_mode is True:
        print('Train:', np.mean(score_train))
        print('Test:', np.mean(score_test))


# Step 12
print(avg_score_LR(kf, X, y))

# from joblib import dump
# dump(avg_score_LR(kf, X, y), 'model.joblib')

# Load the model from the file


# Assuming 'new_data' is your new data for prediction
# Apply the model to make predictions
func_list_test = [
    remove_ids,
    error_mental_preparation,
    error_false_true,
    error_age_group,
    error_athelete_score_negative,
    error_physiotherapy_negative,
    error_athelete_score_over_100,
    error_late_enrolement_0_1,
    error_no_coach_0_1,
    # outlier_high_skewness,
    # missing_categorical_substitute_highest_frequency,
    # missing_numerical_knn_5,
    # missing_numerical_knn_7,
    # missing_drop_all,
    engineer_skewed_data,
    engineer_total_training,
    engineer_region_by_continent,
    # engineer_competition_national_international,
    engineer_competition_local_national_international,
    # engineer_physiotherapy_to_boolean,
    # encoding_true_false_to_1_0,
    encoding_one_hot_encoding_categorical,
    scaling_log_numerical,
    # scaling_sqrt_numerical,
    scaling_numerical_min_max,
    selection_no_coach_drop,
]

df_test = pileline(df_test, func_list_test)

LogisticRegression().fit(X, y)
predictions = model.LogisticRegression(df_test)

# Assuming 'new_data' is your new data for prediction
# Apply the model to make predictions
predictions = model.predict(predictions)

predictions.head()

"""## sem KNN
Train: 0.5964166980619595

Test: 0.5964167072248073

## com KNN 5

Train: 0.735633903798128

Test: 0.7349729896614462

## com KNN 7

Train: 0.7340391412411208

Test: 0.7341691018897331
"""

# Cross Validation

"""Regressão linear
Regressão ligistica

Naive Bayes

Decision tree

Ensemble Algorithms
- 
"""

'''import itertools

list_of_all_functions = list(preprocessing_functions.keys())
print(list_of_all_functions)
print(len(list_of_all_functions))

combination_func_list = []
for r in range(1, len(list_of_all_functions) + 1):
    combination_func_list.extend(list(itertools.combinations(list_of_all_functions, r)))

print(combination_func_list)'''

# Decision Tree

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

func_list_decision_tree = [
    remove_ids,
    error_mental_preparation,
    error_false_true,
    error_age_group,
    error_athelete_score_negative,
    error_physiotherapy_negative,
    error_athelete_score_over_100,
    error_late_enrolement_0_1,
    error_no_coach_0_1,
    # outlier_high_skewness,
    missing_categorical_substitute_highest_frequency,
    # missing_numerical_knn_5,
    missing_numerical_knn_7,
    missing_drop_all,
    engineer_skewed_data,
    engineer_total_training,
    # engineer_region_by_continent,
    # engineer_competition_national_international,
    engineer_competition_local_national_international,
    engineer_physiotherapy_to_boolean,
    # encoding_true_false_to_1_0,
    # encoding_one_hot_encoding_categorical,
    scaling_log_numerical,
    # scaling_sqrt_numerical,
    scaling_numerical_min_max,
    selection_no_coach_drop,
]

func_list_decision_tree = [

]

df_train_decision_tree = np.nan
df_train_decision_tree = df_train.copy()

df_train_decision_tree = pileline(df_train_decision_tree, func_list_decision_tree)

df_train_decision_tree = df_train_decision_tree.select_dtypes(exclude='object')

X = df_train_preprocessed.loc[:, df_train_preprocessed.columns != 'Outcome']
y = df_train_preprocessed['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Create a decision tree classifier
# clf = DecisionTreeClassifier(max_depth=3)
clf = DecisionTreeClassifier(max_leaf_nodes=8)
# clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1", f1score)


def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(4, 10))
    plt.barh(range(n_features), model.feature_importances_, color='yellowgreen')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title('Feature Importance in Decision Tree Classifier')
    plt.show()


plot_feature_importances(clf)

# mostrar todas as funções existentes
# Pronto para copiar par uma lista, até tem a virgula e tudo xD
if test_mode is True:
    for each in list(preprocessing_functions.keys()): print(each + ',')