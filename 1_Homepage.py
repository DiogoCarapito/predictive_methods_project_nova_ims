import streamlit as st
import pandas as pd

# page config
st.set_page_config(
    page_title="PMDM Grupo 11",
    page_icon=":sports_medal:",
    layout="wide")

# page title
st.title("PMDM Grupo 11")

col_1, col_2 = st.columns(2)
with col_1:
    st.markdown("#### Trabalho de grupo de Predictive Methods for Data Mining")
    st.write("""
    - Diogo Carapito (20211202)
    - Joana Alexandre (20211236)
    - Miguel Carvalho (r20181081)
    - Miguel Goulão (20222120)
    - Tomás Santos (20221701)
    """)

with col_2:
    st.image('https://www.portaldalideranca.pt/images/news/Nova-IMS.jpg', width=300)
    
    
st.markdown('[https://github.com/DiogoCarapito/predictive_methods_project_nova_ims](https://github.com/DiogoCarapito/predictive_methods_project_nova_ims)')






