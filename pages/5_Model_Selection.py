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
import datetime as dt
from tqdm import tqdm
import io

# sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


#from itertools import product

st.set_page_config(
    page_title="PMDM Grupo 11",
    page_icon=":sports_medal:",
    layout="wide")

if 'final_predictions' not in st.session_state:
    st.session_state['final_predictions'] = pd.DataFrame(columns=['Outcome'])
list_session_state_variables = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'accuracy_val',
    'precision_val',
    'recall_val',
    'f1_val',
    'lr_solver',
    'knn_n_neighbors',
    'dt_max_depth',
    'dt_max_leaf_nodes',
    'rf_criterion'
    'rf_max_depth',
    'rf_max_leaf_nodes',
    'ensemble',
    'bagging',
    'bagging_dt_criterion',
    'bagging_dt_max_depth',
    'bagging_dt_max_leaf_nodes',
    'bagging_max_leaf_nodes',
    'bagging_rf_criterion',
    'bagging_rf_max_depth',
    'bagging_rf_max_leaf_nodes',
    'boosting',
    'boosting_dt_criterion',
    'boosting_dt_max_depth',
    'boosting_dt_max_leaf_nodes',
    'boosting_max_leaf_nodes',
    'boosting_rf_criterion',
    'boosting_rf_max_depth',
    'boosting_rf_max_leaf_nodes',
    'rf_n_estimators',
    'rf_max_features',
    'rf_min_samples_split',
    'rf_min_samples_leaf',
    'rf_bootstrap',
    'rf_criterion',
    'rf_max_depth',
    'knn_n_neighbors',
    'knn_weights',
    'knn_algorithm',
    'knn_leaf_size',
    'knn_p',
    'knn_metric',
    'knn_metric_params',
    'nn_solver',
    'nn_activation',
    'nn_hidden_layer_sizes',
    'nn_max_iter',
    'nn_learning_rate_init',
    'nn_learning_rate',
    'nn_hidden_layer_1',
    'nn_hidden_layer_2',
    'nn_hidden_layer_3',
    'nn_hidden_layer_4',
    'nn_hidden_layer_5',
    'nn_hidden_layer_6',
    'nn_hidden_layer_7',
    'nn_hidden_layer_8',
    'nn_hidden_layer_9',
    'nn_hidden_layer_10',

]

list_session_state_variables_2 = [

    'num_knn',
    'transform',
    'cross_validation_splits',
    'outlier_treatment',
    'rfe',
    'num_features',

]

for each in list_session_state_variables:
    if each not in st.session_state:
       st.session_state[each] = np.nan


# Page Setup
st.title("Model Selection")

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

st.session_state['model'] = st.radio("Model Selection", (
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "Ensembles",
    "Neural Network"
), index=0, horizontal=True)

st.write("----")
st.subheader("Model Parameters")

if st.session_state['model'] == "Logistic Regression":
   st.session_state['lr_solver'] = st.radio("Solver", ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"), index=0, horizontal=True)
   st.session_state['lr_solver']

elif st.session_state['model'] == "KNN":
   st.session_state['knn_n_neighbors'] = st.slider("Number of neighbors", 1, 40, 5, )

elif st.session_state['model'] == "Naive Bayes":
    pass

elif st.session_state['model'] == "Decision Tree":
   st.session_state['dt_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
   st.session_state['dt_max_depth'] = st.slider("Max depth", 1, 40, 5)
   st.session_state['dt_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)

elif st.session_state['model'] == "Random Forest":
   st.session_state['rf_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
   st.session_state['rf_max_depth'] = st.slider("Max depth", 1, 40, 5)
   st.session_state['rf_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)

elif st.session_state['model'] == "Ensembles":
    st.session_state['ensemble'] = st.radio("Ensemble", ("Bagging", "Boosting"), index=0, horizontal=True)

    if st.session_state['ensemble'] == "Bagging":
        st.session_state['bagging'] = st.radio("Bagging", ("Decision Tree", "Random Forest"), index=0, horizontal=True)
        if st.session_state['bagging'] == "Decision Tree":
            st.session_state['bagging_dt_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
            st.session_state['bagging_dt_max_depth'] = st.slider("Max depth", 1, 40, 5)
            st.session_state['bagging_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)
        elif st.session_state['bagging'] == "Random Forest":
            st.session_state['bagging_rf_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
            st.session_state['bagging_rf_max_depth'] = st.slider("Max depth", 1, 40, 5)
            st.session_state['bagging_rf_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)
    elif st.session_state['ensemble'] == "Boosting":
        st.session_state['boosting'] = st.radio("Boosting", ("Decision Tree", "Random Forest"), index=0, horizontal=True)
        if st.session_state['boosting'] == "Decision Tree":
            st.session_state['boosting_dt_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
            st.session_state['boosting_dt_max_depth'] = st.slider("Max depth", 1, 40, 5)
            st.session_state['boosting_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)
        elif st.session_state['boosting'] == "Random Forest":
            st.session_state['boosting_rf_criterion'] = st.radio("Criterion", ("gini", "entropy"), index=0, horizontal=True)
            st.session_state['boosting_rf_max_depth'] = st.slider("Max depth", 1, 40, 5)
            st.session_state['boosting_rf_max_leaf_nodes'] = st.slider("Max leaf nodes", 1, 40, 5)
    else:
        pass

elif st.session_state['model'] == "Neural Network":
    st.session_state['nn_solver'] = st.radio("Solver", ("lbfgs", "sgd", "adam"), index=0, horizontal=True)
    st.session_state['nn_activation'] = st.radio("Activation", ("logistic", "tanh", "relu"), index=0, horizontal=True)

    col_hl_1, col_hl_2, col_hl_3, col_hl_4, col_hl_5  = st.columns(5)
    with col_hl_1: st.session_state['nn_hidden_layer_1'] = st.number_input("Hidden layer 1 size", 1, 100, 10)
    with col_hl_2: st.session_state['nn_hidden_layer_2'] = st.number_input("Hidden layer 2 size", 0, 100, 0)
    with col_hl_3: st.session_state['nn_hidden_layer_3'] = st.number_input("Hidden layer 3 size", 0, 100, 0)
    with col_hl_4: st.session_state['nn_hidden_layer_4'] = st.number_input("Hidden layer 4 size", 0, 100, 0)
    with col_hl_5: st.session_state['nn_hidden_layer_5'] = st.number_input("Hidden layer 5 size", 0, 100, 0)
    col_hl_6, col_hl_7, col_hl_8, col_hl_9, col_hl_10,  = st.columns(5)
    with col_hl_6: st.session_state['nn_hidden_layer_6'] = st.number_input("Hidden layer 6 size", 0, 100, 0)
    with col_hl_7: st.session_state['nn_hidden_layer_7'] = st.number_input("Hidden layer 7 size", 0, 100, 0)
    with col_hl_8: st.session_state['nn_hidden_layer_8'] = st.number_input("Hidden layer 8 size", 0, 100, 0)
    with col_hl_9: st.session_state['nn_hidden_layer_9'] = st.number_input("Hidden layer 9 size", 0, 100, 0)
    with col_hl_10: st.session_state['nn_hidden_layer_10'] = st.number_input("Hidden layer 10 size", 0, 100, 0)
    st.session_state['nn_hidden_layer_sizes'] = (st.session_state['nn_hidden_layer_1'], st.session_state['nn_hidden_layer_2'], st.session_state['nn_hidden_layer_3'], st.session_state['nn_hidden_layer_4'], st.session_state['nn_hidden_layer_5'], st.session_state['nn_hidden_layer_6'], st.session_state['nn_hidden_layer_7'], st.session_state['nn_hidden_layer_8'], st.session_state['nn_hidden_layer_9'], st.session_state['nn_hidden_layer_10'])
    st.session_state['nn_hidden_layer_sizes'] = list(filter(lambda x: x != 0, st.session_state['nn_hidden_layer_sizes']))

    st.session_state['nn_max_iter'] = st.slider("Max iter", 1, 1000, 200)
    st.session_state['nn_learning_rate_init'] = st.radio("Learning rate init", [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], index=0, horizontal=True)
    st.session_state['nn_learning_rate'] = st.radio("Learning rate", ("constant", "invscaling", "adaptive"), index=0, horizontal=True)

st.write("----")


# MENU
st.subheader("Data Preprocessing Parameters")


st.session_state['num_knn'] = st.radio("Missing Values KNN Number of neighbors", [1, 3, 5, 7, 9, 11, 13, 15], index=2, horizontal=True)
st.session_state['transform'] = st.radio("Transform", ("Logarithm", "Square root", "None"), index=0, horizontal=True)
st.session_state['cross_validation_splits'] = st.radio("Cross Validation Splits", ("2 splits", "5 splits", "10 splits" ), index=1, horizontal=True)
st.session_state['outlier_treatment'] = st.radio("Outlier outlier_treatment", ("negative_to_0", "mod (remover sinal negativo)"), index=0, horizontal=True)
st.session_state['rfe'] = st.checkbox("Recursive Feature Elimination", value=False)
st.session_state['num_features'] = st.slider("Number of features", 1, 40, 10, disabled = bool(not st.session_state['rfe']))


df = df_train.copy()
df_test_copy = df_test.copy()

# PREPROCESSING BEFORE SPLITTING

# 1. remover variaveis_nao_uteis
df = df.drop(variaveis_nao_uteis, axis=1)
df_test_copy = df_test_copy.drop(variaveis_nao_uteis, axis=1)

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
    df_test_copy[column] = df_test_copy[column].replace(substitutes)

# 3. Substituir 0 por 0-35 na variavel Age group
substitute = {
    '0': '0-35',
}
df['Age group'] = df['Age group'].replace(substitute)
df_test_copy['Age group'] = df_test_copy['Age group'].replace(substitute)

# 4. Late enrollment na realidade é um booleano disfarcado de 0 e 1
df['Late enrollment'] = df['Late enrollment'].replace({0: False,1: True})
df_test_copy['Late enrollment'] = df_test_copy['Late enrollment'].replace({0: False,1: True})

# 5. No coach na realidade é um booleano disfarcado de 0 e 1
df['No coach'] = df['No coach'].replace({0: False,1: True})
df_test_copy['No coach'] = df_test_copy['No coach'].replace({0: False,1: True})

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
imputer = KNNImputer(n_neighbors=st.session_state['num_knn'])
df[list_mising_numerical] = imputer.fit_transform(df[list_mising_numerical])


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

df_test_copydf_test_copy = df_test_copy[list_one_hot_encoding]
df_one_hot_encoded_test = pd.get_dummies(df_test_copydf_test_copy, drop_first=True)
df_merged_test = df_test_copy.merge(df_one_hot_encoded_test, left_index=True, right_index=True)
df_test_copy = df_merged_test.drop(list_one_hot_encoding, axis=1)

# 11. True False to 1 0
list_true_false_to_1_0 = list(df.select_dtypes(include='bool').columns)
#st.write(list_true_false_to_1_0)
# iteração sobre variaveis
for variable in list_true_false_to_1_0:
    # substituição de valores acima de 0 para 1
    df[variable] = df[variable].replace({True: 1, False: 0})
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
    'Other training',
    'Total training']
# aplicação do logaritmo
for variable in log_transforms:
    try:
        if st.session_state['transform'] == "Logarithm":
            df[variable] = np.log(df[variable] + 0.01)
            df_test_copy[variable] = np.log(df_test_copy[variable] + 0.01)
        elif st.session_state['transform'] == "Square root":
            df[variable] = np.sqrt(df[variable])
            df_test_copy[variable] = np.sqrt(df_test_copy[variable])
        else:
            pass
    except:
        pass





# 9. Drop all missing values
df = df.dropna()

#st.write("dataframe before split")
#st.write(df.shape[0])
#st.table(df.head(10))
# contar numero de missing values
#st.write("Misisng values",df.isnull().sum().sum())


### SPLIT ###
# X and y
X_df = df.loc[:, df.columns != 'Outcome']
y_df = df['Outcome']

X_df_test_copy = df_test_copy.loc[:, df_test_copy.columns != 'Outcome']


X, X_validation, y, y_validation = train_test_split(
    X_df,
    y_df,
    test_size=0.20,
    random_state=15,
    shuffle=True,
    stratify=y_df
)



accuracy = []
precision = []
recall = []
f1 = []

accuracy_val = []
precision_val = []
recall_val = []
f1_val = []

if 'model_run' not in st.session_state:
    st.session_state['model_run'] = False

st.button("Run Model", type="primary", on_click=lambda: st.session_state.update(model_run=True))

if st.session_state['model_run']:
    st.write("----")
    st.write("Running model...")
    progress_bar = st.progress(0)
    progress = 0

    # CROSS VALIDATION
    if st.session_state['cross_validation_splits'] == "2 splits":
        num_splits = 2
    elif st.session_state['cross_validation_splits'] == "5 splits":
        num_splits = 5
    else:
        num_splits = 10

    # method: KFold
    for train_index, test_index in tqdm(KFold(n_splits=num_splits).split(X)):
        progress += 1 / num_splits
        progress_bar.progress(progress)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_val = X_validation.copy()
        X_df_test_copy = df_test_copy.copy()

        # PREPROCESSING AFTER SPLITTING
        # OUTLIERS
        if st.session_state['outlier_treatment'] == 'negative_to_0':

            # 1. substituir athlet score < 0 por 0
            X_train['Athlete score'] = X_train['Athlete score'].apply(lambda x: 0 if x < 0 else x).astype(float)

            # 2. substituir physiotherapy < 0 por 0
            X_train['Physiotherapy'] = X_train['Physiotherapy'].apply(lambda x: 0 if x < 0 else x).astype(float)

        else:
            # 1. substituir athlet score < 0 por pelo modulo abs()
            X_train['Athlete score'] = X_train['Athlete score'].apply(abs)

            # 2. substituir physiotherapy < 0 por pelo modulo abs()
            X_train['Physiotherapy'] = X_train['Physiotherapy'].apply(abs)

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
            # substituição de tudo o que for superior a 1 passar a 1
            X_train[each] = X_train[each].apply(lambda x: 1 if x > 1 else x)

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
        minmax_train = pd.DataFrame(
            minmax_train,
            columns = df_numericas.columns,
            index = df_numericas.index)
        # Substituir valores no dataframe original para novos valores MinMax
        X_train[variaveis_numericas] = minmax_train

        minmax_test = minmax_scaler.transform(X_test[variaveis_numericas])
        minmax_test = pd.DataFrame(
            minmax_test,
            columns = X_test[variaveis_numericas].columns,
            index = X_test[variaveis_numericas].index)
        X_test[variaveis_numericas] = minmax_test

        #st.write(X_val.columns)
        minmax_val = minmax_scaler.transform(X_val[variaveis_numericas])
        minmax_val = pd.DataFrame(
            minmax_val,
            columns = X_val[variaveis_numericas].columns,
            index = X_val[variaveis_numericas].index)
        X_val[variaveis_numericas] = minmax_val

        #st.write(X_df_test_copy.columns)
        #st.table(X_df_test_copy.head(3))
        X_minmax_df_test_copy = minmax_scaler.transform(X_df_test_copy[variaveis_numericas])
        #st.write(X_minmax_df_test_copy)
        X_minmax_df_test_copy = pd.DataFrame(
            X_minmax_df_test_copy,
            columns=X_df_test_copy[variaveis_numericas].columns,
            index=X_df_test_copy[variaveis_numericas].index)
        X_df_test_copy[variaveis_numericas] = X_minmax_df_test_copy


        # RFE
        # 15. Recursive Feature Elimination
        if st.session_state['rfe'] is True:
            rfe_model = LogisticRegression()
            rfe = RFE(estimator=rfe_model, n_features_to_select=st.session_state['num_features'])
            X_rfe = rfe.fit_transform(X=X_train, y=y_train)
            selected_features = pd.Series(rfe.support_, index=X_train.columns)
            X_test = X_test.loc[:, selected_features]
            X_train = X_train.loc[:, selected_features]
            X_val = X_val.loc[:, selected_features]
            X_df_test_copy = X_df_test_copy.loc[:, selected_features]
            #st.write(list(selected_features))

        # run the model
        #st.table(X_train.head())
        if st.session_state['model'] == "Logistic Regression":
            model = LogisticRegression(
                solver = st.session_state['lr_solver']
            ).fit(X_train, y_train)
        elif st.session_state['model'] == "Random Forest":
            model = RandomForestClassifier(
                criterion = st.session_state['rf_criterion'],
                max_depth = st.session_state['rf_max_depth'],
                max_leaf_nodes = st.session_state['rf_max_leaf_nodes'],
            ).fit(X_train, y_train)
        elif st.session_state['model'] == "Decision Tree":
            model = DecisionTreeClassifier(
                max_leaf_nodes = st.session_state['dt_max_leaf_nodes'],
                max_depth = st.session_state['dt_max_depth'],
                criterion = st.session_state['dt_criterion'],
            ).fit(X_train, y_train)
        elif st.session_state['model'] == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=st.session_state['knn_n_neighbors']
            ).fit(X_train, y_train)
        elif st.session_state['model'] == "Ensambles":
            model = VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression()),
                    ('rf', RandomForestClassifier()),
                    ('dt', DecisionTreeClassifier()),
                    ('knn', KNeighborsClassifier()),
                    ('svm', LinearSVC()),
                    ('nb', GaussianNB()),
                    ('xgb', XGBClassifier()),
                    ('mlp', MLPClassifier())
                ],
                voting='hard'
            ).fit(X_train, y_train)
        elif st.session_state['model'] == "SVM":
            model = LinearSVC().fit(X_train, y_train)
        elif st.session_state['model'] == "Naive Bayes":
            model = GaussianNB().fit(X_train, y_train)
        elif st.session_state['model'] == "XGBoost":
            model = XGBClassifier().fit(X_train, y_train)
        elif st.session_state['model'] == "Neural Network":
            model = MLPClassifier(
                hidden_layer_sizes = st.session_state['nn_hidden_layer_sizes'],
                activation = st.session_state['nn_activation'],
                solver = st.session_state['nn_solver'],
                max_iter=st.session_state['nn_max_iter'],
                learning_rate=st.session_state['nn_learning_rate'],
                learning_rate_init=st.session_state['nn_learning_rate_init'],
            ).fit(X_train, y_train)

        predictions = model.predict(X_test)


        accuracy.append(accuracy_score(y_test, predictions))
        precision.append(precision_score(y_test, predictions))
        recall.append(recall_score(y_test, predictions))
        f1.append(f1_score(y_test, predictions))

        predictions_val = model.predict(X_val)
        accuracy_val.append(accuracy_score(y_validation, predictions_val))
        precision_val.append(precision_score(y_validation, predictions_val))
        recall_val.append(recall_score(y_validation, predictions_val))
        f1_val.append(f1_score(y_validation, predictions_val))

    progress_bar.progress(100)


    st.session_state['final_predictions'] = model.predict(X_df_test_copy)
    st.session_state['final_predictions'] = pd.DataFrame(st.session_state['final_predictions'], index=X_df_test_copy.index, columns=['Outcome'])

    st.session_state['accuracy'] = round(100*np.mean(accuracy),2)
    st.session_state['precision'] = round(100*np.mean(precision),2)
    st.session_state['recall'] = round(100*np.mean(recall),2)
    st.session_state['f1'] = round(100*np.mean(f1),2)



    st.session_state['accuracy_val'] = round(100*np.mean(accuracy_val),2)
    st.session_state['precision_val'] = round(100*np.mean(precision_val),2)
    st.session_state['recall_val'] = round(100*np.mean(recall_val),2)
    st.session_state['f1_val'] = round(100*np.mean(f1_val),2)


    df_best = pd.read_csv('model_performance_records.csv')
    df_best = df_best[df_best['F1']==df_best['F1'].max()]


    st.subheader('Test Performance')
    try:
        col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
        with col_metrics_1:
            st.metric('Accuracy', str(st.session_state['accuracy'])+'%', st.session_state['accuracy']-df_best['Accuracy'].values[0])
        with col_metrics_2:
            st.metric('Precision', str(st.session_state['precision'])+'%', st.session_state['precision']-df_best['Precision'].values[0])
        with col_metrics_3:
            st.metric('Recall', str(st.session_state['recall'])+'%', st.session_state['recall']-df_best['Recall'].values[0])
        with col_metrics_4:
            st.metric('F1', str(st.session_state['f1'])+'%', st.session_state['f1']-df_best['F1'].values[0])
    except:
        col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
        with col_metrics_1:
            st.metric('Accuracy', str(st.session_state['accuracy']) + '%')
        with col_metrics_2:
            st.metric('Precision', str(st.session_state['precision']) + '%')
        with col_metrics_3:
            st.metric('Recall', str(st.session_state['recall']) + '%')
        with col_metrics_4:
            st.metric('F1', str(st.session_state['f1']) + '%')
    st.subheader('Validation Performance')
    col_metrics_val_1, col_metrics_val_2, col_metrics_val_3, col_metrics_val_4 = st.columns(4)
    with col_metrics_val_1:
        st.metric('Accuracy', str(st.session_state['accuracy_val']) + '%', str(round(st.session_state['accuracy_val']-st.session_state['accuracy'], 2))+"%")
    with col_metrics_val_2:
        st.metric('Precision', str(st.session_state['precision_val']) + '%', str(round(st.session_state['precision_val']-st.session_state['precision'], 2))+"%")
    with col_metrics_val_3:
        st.metric('Recall', str(st.session_state['recall_val']) + '%', str(round(st.session_state['recall_val']-st.session_state['recall'], 2))+"%")
    with col_metrics_val_4:
        st.metric('F1', str(st.session_state['f1_val']) + '%', str(round(st.session_state['f1_val']-st.session_state['f1'], 2))+"%")


    st.write("----")

    col_outcome_1, col_outcome_2 = st.columns(2)
    with col_outcome_1:
        st.metric("Number of Outcome = 1", str(len(st.session_state['final_predictions'][st.session_state['final_predictions']['Outcome']==1])))
    with col_outcome_2:
        st.metric("Number of Outcome = 0", str(len(st.session_state['final_predictions'][st.session_state['final_predictions']['Outcome']==0])))


    sv_data = st.session_state['final_predictions'].to_csv(index=True)
    buffer = io.BytesIO()
    buffer.write(sv_data.encode())
    buffer.seek(0)

model_performance_record = pd.DataFrame({
    'Date': [dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
    'Model': [st.session_state['model']],
    'Accuracy': st.session_state['accuracy'],
    'Precision': st.session_state['precision'],
    'Recall': st.session_state['recall'],
    'F1': st.session_state['f1'],
    'Accuracy_val': st.session_state['accuracy_val'],
    'Precision_val': st.session_state['precision_val'],
    'Recall_val': st.session_state['recall_val'],
    'F1_val': st.session_state['f1_val'],
    'Number of Outcome = 1': len(st.session_state['final_predictions'][st.session_state['final_predictions']['Outcome']==1]),
    'Number of Outcome = 0': len(st.session_state['final_predictions'][st.session_state['final_predictions']['Outcome']==0]),

    'num_knn': st.session_state['num_knn'],
    'transform': st.session_state['transform'],
    'cross_validation_splits': st.session_state['cross_validation_splits'],
    'outlier_treatment': st.session_state['outlier_treatment'],
    'rfe': st.session_state['rfe'],
    'num_features': st.session_state['num_features'],

    'lr_solver': st.session_state['lr_solver'],

    'knn_n_neighbors': st.session_state['knn_n_neighbors'],

    'dt_max_depth': st.session_state['dt_max_depth'],
    'dt_max_leaf_nodes': st.session_state['dt_max_leaf_nodes'],

    'rf_criterion': st.session_state['rf_criterion'],
    'rf_max_depth': st.session_state['rf_max_depth'],
    'rf_max_leaf_nodes': st.session_state['rf_max_leaf_nodes'],

    'ensemble': st.session_state['ensemble'],
    'bagging': st.session_state['bagging'],
    'bagging_dt_criterion': st.session_state['bagging_dt_criterion'],
    'bagging_dt_max_depth': st.session_state['bagging_dt_max_depth'],
    'bagging_max_leaf_nodes': st.session_state['bagging_max_leaf_nodes'],
    'bagging_rf_criterion': st.session_state['bagging_rf_criterion'],
    'bagging_rf_max_depth': st.session_state['bagging_rf_max_depth'],
    'bagging_rf_max_leaf_nodes': st.session_state['bagging_rf_max_leaf_nodes'],
    'boosting': st.session_state['boosting'],
    'boosting_dt_criterion': st.session_state['boosting_dt_criterion'],
    'boosting_dt_max_depth': st.session_state['boosting_dt_max_depth'],
    'boosting_max_leaf_nodes': st.session_state['boosting_max_leaf_nodes'],
    'boosting_rf_criterion': st.session_state['boosting_rf_criterion'],
    'boosting_rf_max_depth': st.session_state['boosting_rf_max_depth'],
    'boosting_rf_max_leaf_nodes': st.session_state['boosting_rf_max_leaf_nodes'],

    'nn_solver': st.session_state['nn_solver'],
    'nn_activation': st.session_state['nn_activation'],
    #'nn_hidden_layer_sizes': st.session_state['nn_hidden_layer_sizes'],
    'nn_max_iter': st.session_state['nn_max_iter'],
    'nn_learning_rate_init': st.session_state['nn_learning_rate_init'],
    'nn_learning_rate': st.session_state['nn_learning_rate'],
    'nn_hidden_layer_1': st.session_state['nn_hidden_layer_1'],
    'nn_hidden_layer_2': st.session_state['nn_hidden_layer_2'],
    'nn_hidden_layer_3': st.session_state['nn_hidden_layer_3'],
    'nn_hidden_layer_4': st.session_state['nn_hidden_layer_4'],
    'nn_hidden_layer_5': st.session_state['nn_hidden_layer_5'],
    'nn_hidden_layer_6': st.session_state['nn_hidden_layer_6'],
    'nn_hidden_layer_7': st.session_state['nn_hidden_layer_7'],
    'nn_hidden_layer_8': st.session_state['nn_hidden_layer_8'],
    'nn_hidden_layer_9': st.session_state['nn_hidden_layer_9'],
    'nn_hidden_layer_10': st.session_state['nn_hidden_layer_10'],

})

if model_performance_record['Accuracy'] is not np.nan:
    with open('model_performance_records.csv', mode='a') as file:
        model_performance_record.to_csv(file, header=file.tell() == 0, index=False)

st.session_state['model_run'] = False
st.write('----')

try:
    st.download_button(label='Download predictions', data=sv_data, file_name='solution.csv', disabled=st.session_state['model_run'])
except:
    pass

