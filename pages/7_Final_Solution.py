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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


st.title("Final Solution")
st.write("---")

# Data Loading
st.title("Data Loading")
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')

# Colocar o RecordID como index
df_train.set_index('RecordID', inplace=True)
df_train.index.name = 'RecordID'
df_test.set_index('RecordID', inplace=True)

# Mostrar dados em tabelas
st.header("train.csv")
st.dataframe(df_train.head())
st.header("test.csv")
st.dataframe(df_test.head())

# Definição de tipos de variáveis
# Target
target = 'Outcome'

# Variáveis não úteis
variaveis_nao_uteis = ['Athlete Id']

# Variáveis numéricas
variaveis_numericas = list(df_train.drop(variaveis_nao_uteis, axis=1).select_dtypes(include=['int', 'float']).columns)
variaveis_numericas.remove(target)

# Variáveis booleanas
variaveis_booleanas = [
    'Disability',
    'Late enrollment',
    'Cancelled enrollment',
    'Mental preparation',
    'Outdoor Workout',
    'No coach',
    'Past injuries'
]

# Categóricas
variaveis_categoricas = [
    'Competition',
    'Sex',
    'Region',
    'Education',
    'Age group',
    'Income'
]

# Data preprocessing
st.title("Data Preprocessing")

# 1. remover variaveis_nao_uteis
df_train = df_train.drop(variaveis_nao_uteis, axis=1)
df_test = df_test.drop(variaveis_nao_uteis, axis=1)

# ERROS
# 2. substituir errados por False e True
substitutes = {
    'FASE': False,
    'FALSE': False,
    'TRUE': True,
}
columns_to_replace = [
    'Disability',
    'Late enrollment',
    'Cancelled enrollment',
    'Mental preparation',
    'Outdoor Workout',
    'No coach',
    'Past injuries'
]
for column in columns_to_replace:
    df_train[column] = df_train[column].replace(substitutes)
    df_test[column] = df_test[column].replace(substitutes)


# 3. Substituir 0 por 0-35 na váriavel Age group
substitute = {
    '0': '0-35',
}
df_train['Age group'] = df_train['Age group'].replace(substitute)
df_test['Age group'] = df_test['Age group'].replace(substitute)

# 4. Late enrollment na realidade é um booleano disfarcado de 0 e 1
df_train['Late enrollment'] = df_train['Late enrollment'].replace({0: False,1: True})
df_test['Late enrollment'] = df_test['Late enrollment'].replace({0: False,1: True})

# 5. No coach na realidade é um booleano disfarcado de 0 e 1
df_train['No coach'] = df_train['No coach'].replace({0: False,1: True})
df_test['No coach'] = df_test['No coach'].replace({0: False,1: True})


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
    df_train[variavel] = df_train[variavel].fillna(valor_sub)

# 8. missing values por KNN = 9
list_mising_numerical = [
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
imputer = KNNImputer(n_neighbors=9)
df_train[list_mising_numerical] = imputer.fit_transform(df_train[list_mising_numerical])


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
df_one_hot_encoding = df_train[list_one_hot_encoding]
# one hot encoding, eliminando a primeira coluna de forma a não dar erro
df_one_hot_encoded = pd.get_dummies(df_one_hot_encoding, drop_first=True)
# Junção das novas colunas com o dataframe original
df_merged = df_train.merge(df_one_hot_encoded, left_index=True, right_index=True)
# remoção das colunas antigas
df_train = df_merged.drop(list_one_hot_encoding, axis=1)

# Fazer o mesmo para o df_test
df_test_hot_encoding = df_test[list_one_hot_encoding]
df_test_hot_encoding = pd.get_dummies(df_test_hot_encoding, drop_first=True)
df_merged_test = df_test.merge(df_test_hot_encoding, left_index=True, right_index=True)
df_test = df_merged_test.drop(list_one_hot_encoding, axis=1)


# 11. True False to 1 0
#list_true_false_to_1_0 = list(df.select_dtypes(include='bool').columns)
list_true_false_to_1_0 = variaveis_booleanas
# iteração sobre variaveis
for variable in list_true_false_to_1_0:
    # substituição de valores acima de 0 para 1
    df_train[variable] = df_train[variable].replace({True: 1, False: 0})
    df_test_copy[variable] = df_test_copy[variable].replace({True: 1, False: 0})



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
    'Other training',]
# aplicação do logaritmo

for variable in log_transforms:
    df_train[variable] = np.log(df_train[variable] + 0.01)
    df_test[variable] = np.log(df_test[variable] + 0.01)

# 9. Drop all missing values
df_train = df_train.dropna()

#st.dataframe(df_train.describe().T)
#st.dataframe(df_test.describe().T)


#SPLIT
X_df_train = df_train.loc[:, df_train.columns != 'Outcome']
y_df_train = df_train['Outcome']

X_df_test = df_test.loc[:, df_test.columns != 'Outcome']


X, X_validation, y, y_validation = train_test_split(
    X_df_train,
    y_df_train,
    test_size=0.20,
    random_state=15,
    shuffle=True,
    stratify=y_df_train
)

accuracy = []
precision = []
recall = []
f1 = []

accuracy_val = []
precision_val = []
recall_val = []
f1_val = []

# CROSS VALIDATION
num_splits = 5

# method: KFold
for train_index, test_index in KFold(n_splits=num_splits).split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_val = X_validation.copy()
    #X_df_test_copy = df_test_copy.copy()

    # PREPROCESSING AFTER SPLITTING
    # OUTLIERS

    # 1. substituir athlet score < 0 por 0
    X_train['Athlete score'] = X_train['Athlete score'].apply(lambda x: 0 if x < 0 else x).astype(float)

    # 2. substituir physiotherapy < 0 por 0
    X_train['Physiotherapy'] = X_train['Physiotherapy'].apply(lambda x: 0 if x < 0 else x).astype(float)

        # Aletrtiva outliers
        # 1. substituir athlet score < 0 por pelo modulo abs()
        #X_train['Athlete score'] = X_train['Athlete score'].apply(abs)

        # 2. substituir physiotherapy < 0 por pelo modulo abs()
        #X_train['Physiotherapy'] = X_train['Physiotherapy'].apply(abs)

    # 3. substituir athlete score > 100 por 100
    X_train['Athlete score'] = X_train['Athlete score'].apply(lambda x: 100 if x > 100 else x).astype(float)

    # 4. substituir valores de skewness muito elevados por um tecto maximo
    #high_skewness_variables = {
        #'Sand training': 500,
        #'Recovery': 4000,
        #'Cardiovascular training': 4000,
        #'Squad training': 150,
        #'Physiotherapy': 800,
        #'Sport-specific training': 320,
        #'Other training': 130,
    #}
    #for key, value in high_skewness_variables.items():
        #X_train[key] = X_train[key].apply(lambda x: value if x > value else x).astype(float)

    # 13. substituir valores de skewness muito elevados por um valor mais pequeno
    skewed_data = [
        'Previous attempts',
        'Sand training',
        'Plyometric training',
        'Other training', ]
    # iteração pelo skewed data
    for each in skewed_data:
        # substituição de tudo o que for superior a 1 passar a 1
        X_train[each] = X_train[each].apply(lambda x: 1 if x > 1 else x)