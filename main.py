import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

pickle_in = open('Diabetes.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict():
    st.sidebar.header('Birth Genomics Co.')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Congenital Disabilities Prediction (Only for Females Above 19 Years of Age)')
    st.markdown('This trained dataset is originally collected by BirthGenomics Company.')
    st.markdown('We will be very greatful if you share your data in this regard with us.')

    name = st.text_input("DNA Sequence:")
    pregnancy = st.number_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.number_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.number_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.number_input("Please connect the device to your computer, Data Reading from the Devics:")
    st.markdown('Number of time periods consideration in the plot (read from device)')


    age = st.number_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')


    if submit:
        prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!', name,'It seems you are not a risky case')
        else:
            st.write(name,", It seems like you are in risk. But don't lose hope, we strongly recommnd visit a especialist doctor")


def main():
    new_title = '<p style="font-size: 42px;">Welcome The Congenital Disabilities Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit  
    to demonstrate Congenital Disabilities Prediction. It performs prediction on multiple parameters
                                  """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Predict Congenital Disabilities"))
    if choice == "Predict Congenital Disabilities":
        read_me_0.empty()
        read_me.empty()
        predict()
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
    
