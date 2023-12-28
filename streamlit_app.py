import pickle

import streamlit as st

st.title("Number Input Example")

number_input = st.number_input("Enter a number:", min_value=0.0, max_value=100.0, step=0.1)

st.write("You entered:", number_input)

with open('svm_model.pkl', 'rb') as model_file:

    loaded_model = pickle.load(model_file)
