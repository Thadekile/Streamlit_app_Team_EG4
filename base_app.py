import streamlit as st
import pickle

# Load your model
with open('logistic_regression_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# App Title
st.title('Text Prediction App')

# User Input
st.header('Enter Text:')
user_text = st.text_area('Type or paste your text here', height=200)

# Prediction Button
if st.button('Predict'):
    if user_text:  # Check if text is provided

        # Preprocess Text (Optional but often needed)
        # ... Apply any necessary preprocessing steps to the text (cleaning, tokenization, etc.)

        # Make Prediction
        prediction = model.predict([user_text])[0]

        # Display Prediction
        st.subheader('Prediction:')
        st.write(prediction)

    else:
        st.warning('Please enter some text.')
