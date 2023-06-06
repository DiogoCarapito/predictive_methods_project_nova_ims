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

st.title("Sandbox")


full_path = path + "train.csv"

# load dataset
df = pd.read_csv(full_path)

st.table(df[df['Physiotherapy'] <0 ])
st.write(df[df['Physiotherapy'] <0 ].shape[0])

