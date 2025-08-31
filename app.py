import streamlit as st
import pandas as pd
import joblib  # or pickle to load your saved model

# Load your trained model
model = joblib.load('rainfall_model.pkl')

st.title('Rainfall Prediction Demo')

# Input fields example (adjust per your features)
location = st.selectbox('Location', ['Albury', 'Sydney', 'Melbourne', 'Brisbane'])
min_temp = st.number_input('Min Temperature')
max_temp = st.number_input('Max Temperature')
rainfall = st.number_input('Rainfall')

# When user clicks Predict button
if st.button('Predict Rain Tomorrow'):
    # Create DataFrame from inputs
    input_df = pd.DataFrame({
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        # add other features as needed
    })

    # Preprocess input_df as your pipeline requires

    prediction = model.predict(input_df)[0]
    result = 'Yes' if prediction == 1 else 'No'

    st.write(f'Will it rain tomorrow? **{result}**')
