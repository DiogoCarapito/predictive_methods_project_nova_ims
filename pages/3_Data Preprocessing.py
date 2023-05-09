import streamlit as st
import pandas as pd

#import preproc
#all_functions = inspect.getmembers(preproc, inspect.isfunction)

st.title("Data Preprocessing")

# Data Input

with st.form(key='preprocessing_form'):
    st.write("### Preprocessing")

    st.write("#### Errors")

for each in all_functions:
    st.write(each[0])


