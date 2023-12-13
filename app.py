import pickle
import numpy as np
import streamlit as st
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


st.title("Diabetes Prediction System")

pregnencies = st.slider('Pregnencies', 0, 15,0)
glucose = st.slider('Glucose Level', 0, 200,120)
bp = st.slider('Blood Pressure', 0, 125,70)
skin = st.slider('Skin Thickness', 0, 100,20)
insulin= st.slider('Insulin Levels', 0, 850,80)
bmi= st.slider('BMI', 0, 70,31)
age= st.slider('Age', 0, 100,30)

if st.button('Predict',type="primary"):
    input_data= (pregnencies,glucose,bp,skin,insulin,bmi,age)
    input_data_as_numpy = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy.reshape(1,-1)
    scaled_input = scaler.transform(input_data_reshaped)
    prediction = model.predict(scaled_input)
    if prediction[0]==0:
        st.write("You Probably Do Not Have Diabetes.")
    else:
        st.write("You May Have Diabetes.")