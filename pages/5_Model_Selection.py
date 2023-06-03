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
import datetime as dt

# sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, \
    mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

#from itertools import product

st.set_page_config(
    page_title="PMDM Grupo 11",
    page_icon=":sports_medal:",
    layout="wide")

# Page Setup
st.title("Model Selection")

# Tabs
tab_logistic, tab_bayes, tab_decision_tree, tab_ensambles, tab_neural_network =  st.tabs(['LogisticRegression', 'Bayes', 'DecisionTree', 'Ensembles', 'NeuralNetwork'])

# Data Loading
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')

df_train.set_index('RecordID', inplace=True)
df_train.index.name = 'RecordID'

df_test.set_index('RecordID', inplace=True)
df_train.index.name = 'RecordID'

# Lista de variáveis
target = 'Outcome'

variaveis_nao_uteis = ['Athlete Id']

variaveis_numericas = list(df_train.drop(variaveis_nao_uteis, axis=1).select_dtypes(include=['int', 'float']).columns)
variaveis_numericas.remove(target)


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



# Tabs para cada modelo
with tab_logistic:
    st.header("Logistic Regression")

    df = df_train.copy()

    # PREPROCESSING BEFORE SPLITTING

    # 1. remover variaveis_nao_uteis
    df = df.drop(variaveis_nao_uteis, axis=1)

    # ERROS
    # 2. substituir errados por False e True
    substitutes = {
        'FASE': False,
        'FALSE': False,
        'TRUE': True,
    }
    columns_to_replace = [
        #'Mental preparation'
        'Disability',
        'Late enrollment',
        'Cancelled enrollment',
        'Mental preparation',
        'Outdoor Workout',
        'No coach',
        'Past injuries'
    ]
    for column in columns_to_replace:
        df[column] = df[column].replace(substitutes)

    # 3. Substituir 0 por 0-35 na variavel Age group
    substitute = {
        '0': '0-35',
    }
    df['Age group'] = df['Age group'].replace(substitute)

    # 4. Late enrollment na realidade é um booleano disfarcado de 0 e 1
    df['Late enrollment'] = df['Late enrollment'].replace({0: False,1: True})

    # 5. No coach na realidade é um booleano disfarcado de 0 e 1
    df['No coach'] = df['No coach'].replace({0: False,1: True})

    # MISSING VAULES
    # 6. missing values pela moda em variaves categoricas
    dict_variavel_sub = {
        'Age group': '0-35',
        'Disability': False,
        'Late enrollment': False,
        'Cancelled enrollment': False,
        'Mental preparation': False,
        'Outdoor Workout': False,
        'No coach': False,
    }
    for variavel, valor_sub in dict_variavel_sub.items():
        df[variavel] = df[variavel].fillna(valor_sub)

    # 8. missing values por KNN = 5
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


    # ENCODING
    # 10. One hot encoding
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
    df = df_merged.drop(list_one_hot_encoding, axis=1)

    # 11. True False to 1 0
    list_true_false_to_1_0 = list(df.select_dtypes(include='bool').columns)
    #st.write(list_true_false_to_1_0)
    # iteração sobre variaveis
    for variable in list_true_false_to_1_0:
        # substituição de valores acima de 0 para 1
        df[variable] = df[variable].replace({True: 1, False: 0})


    # SCALING
    # 12. Transformação de variáveis para o logaritmo para tratar skewness
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
    # aplicação do logaritmo
    for variable in log_transforms:
        try:
            df[variable] = np.log(df[variable] + 0.01)
        except:
            pass


    # 9. Drop all missing values
    df = df.dropna()

    st.write("dataframe before split")
    st.write(df.shape[0])
    st.table(df.head(10))
    # contar numero de missing values
    st.write("Misisng values",df.isnull().sum().sum())




    ### SPLIT ###
    # X and y
    X = df.loc[:, df.columns != 'Outcome']
    y = df['Outcome']


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)





    # PREPROCESSING AFTER SPLITTING
    # OUTLIERS
    # 1. substituir athlet score < 0 por 0
    X_train['Athlete score'] = X_train['Athlete score'].apply(lambda x: 0 if x < 0 else x).astype(float)

    # 2. substituir physiotherapy < 0 por 0
    X_train['Physiotherapy'] = X_train['Physiotherapy'].apply(lambda x: 0 if x < 0 else x).astype(float)

    # 3. substituir athlete score > 100 por 100
    X_train['Athlete score'] = X_train['Athlete score'].apply(lambda x: 100 if x > 100 else x).astype(float)

    # 4. substituir valores de skewness muito elevados por um tecto maximo
    high_skewness_variables = {
        'Sand training': 500,
        'Recovery': 4000,
        'Cardiovascular training': 4000,
        'Squad training': 150,
        'Physiotherapy': 800,
        'Sport-specific training': 320,
        'Other training': 130,
    }
    for key, value in high_skewness_variables.items():
        X_train[key] = X_train[key].apply(lambda x: value if x > value else x).astype(float)

    # 13. substituir valores de skewness muito elevados por um valor mais pequeno
    skewed_data = [
        'Previous attempts',
        'Sand training',
        'Plyometric training',
        'Other training', ]
    # iteração pelo skewed data
    for each in skewed_data:
        # substituição de tudo o que for superior a 0 passar a 1
        df[each] = df[each].apply(lambda x: 1 if x > 0 else x)

    # NORMALIZAÇÂO
    # 14. mimmax scaler
    minmax_scaler = MinMaxScaler()
    # filtrar dataframe apenas pelas variáveis numéricas
    variaveis_numericas = X_train.select_dtypes(include='number').columns.tolist()

    df_numericas = X_train[variaveis_numericas]
    # execução do MinMax
    minmax_scaler = minmax_scaler.fit(df_numericas)
    minmax_train = minmax_scaler.transform(df_numericas)

    # Transforma resultado num dataframe
    minmax_train = pd.DataFrame(minmax_train, columns=df_numericas.columns, index=df_numericas.index)

    # Substituir valores no dataframe original para novos valores MinMax
    X_train[variaveis_numericas] = minmax_train


    minmax_test = minmax_scaler.transform(X_test[variaveis_numericas])
    minmax_test = pd.DataFrame(minmax_test, columns=X_test[variaveis_numericas].columns, index=X_test[variaveis_numericas].index)
    X_test[variaveis_numericas] = minmax_test


    X_train = X_train.select_dtypes(include='number')
    X_test = X_test.select_dtypes(include='number')
    #st.write(X_train.head(5))


    st.write("X_train shape:", X_train.shape[0])
    st.table(X_train.head(10))
    st.write("X_test shape:", X_test.shape[0])
    st.table(X_test.head(10))

    # model training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # model evaluation
    predictions = model.predict(X_test)

    accuracy = round(100*accuracy_score(y_test, predictions),2)
    precision = round(100*precision_score(y_test, predictions),2)
    recall = round(100*recall_score(y_test, predictions),2)
    f1 = round(100*f1_score(y_test, predictions),2)

    conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

    st.subheader('Model Performance')
    col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
    with col_metrics_1: st.metric('Accuracy', str(accuracy)+'%')
    with col_metrics_2: st.metric('Precision', str(precision)+'%')
    with col_metrics_3: st.metric('Recall', str(recall)+'%')
    with col_metrics_4: st.metric('F1', str(f1)+'%')


    st.subheader('Confusion Matrix')
    st.table(conf_matrix)

    if st.button('Save prediction'):
        final_prediction = model.predict(df_test)
        st.table(df_train)
        # save in a csv file
        with open('predictions_logistic_regression.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['RecordID','Outcome'])
            for row in final_prediction:
                writer.writerow(row)
        st.success('Saved')


with tab_bayes:
    st.header("Bayes")


with tab_decision_tree:
    st.header("Decision Tree")

with tab_ensambles:
    st.header("Ensembles Methods")

with tab_neural_network:
    st.header("Neural Networks")

