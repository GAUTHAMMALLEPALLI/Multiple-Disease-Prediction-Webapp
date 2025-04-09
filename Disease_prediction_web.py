# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:57:46 2025

@author: mallepalli gautham
"""

import numpy as np
import pickle
import streamlit as st

#load the model
loaded_model=pickle.load(open('D:/project\GAUTHAM_PRO1/trained_model.sav','rb'))

def diabetes_prediction(input_symptoms):
    
    # Define symptom list (all feature names)
    all_symptoms = ["Fever", "Shortness of breath", "Cough", "Chest pain", "Nausea", "Vomiting", "Lightheadedness", "Sweating", "Sudden weakness", "Numbness", "Confusion", "Headache", "Lump", "Weight loss", "Fatigue", "Bleeding", "Seizures", "Swelling", "Conjunctivitis", "Diarrhea", "Liver Damage", "Cancer", "Stiff Neck", "Pain in upper abdomen"]
    
    # Converting the input to feature vector
    input_data = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]

    #Changing the data into numpy array data frame
    input_data_as_numpy=np.asarray(input_data)

    #Reshaping the data
    input_data_reshaped=input_data_as_numpy.reshape(1,-1)

    #Predicting the Output
    prediction=loaded_model.predict(input_data_reshaped)

    #Printing the output disesase name
    return f"The person is having {prediction[0]} disease"

    
  
def main():
    # Giving the title
    st.title('Multiple Disease Prediction System by Gautham Mallepalli')

    # Define symptom list (all feature names)
    all_symptoms = ["NA","Fever", "Shortness of breath", "Cough", "Chest pain", "Nausea", "Vomiting", "Lightheadedness", "Sweating", "Sudden weakness", "Numbness", "Confusion", "Headache", "Lump", "Weight loss", "Fatigue", "Bleeding", "Seizures", "Swelling", "Conjunctivitis", "Diarrhea", "Liver Damage", "Cancer", "Stiff Neck", "Pain in upper abdomen"]

    # Select 5 symptoms from the list
    symptom1 = st.selectbox("Select Symptom 1", all_symptoms)
    symptom2 = st.selectbox("Select Symptom 2", all_symptoms)
    symptom3 = st.selectbox("Select Symptom 3", all_symptoms)
    symptom4 = st.selectbox("Select Symptom 4", all_symptoms)
    symptom5 = st.selectbox("Select Symptom 5", all_symptoms)

    # Create a list of selected symptoms
    selected_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

    # Creating a null string to store output
    diagnosis = ''

    # Creating button for prediction
    if st.button('Predict Disease'):
        diagnosis = diabetes_prediction(selected_symptoms)

    # Displaying output
    st.success(diagnosis)


if __name__ == '__main__':
    main()
