import streamlit as st
import pickle

# Model Options
model_options = {
    "Logistic Regression": "models/logistic_regression_pipeline.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "KNN Classifier": "models/knn_pipeline.pkl"
}

# App Title
st.title('Story Stream')
st.write("This app will take in some text data you have, and predict the category this belongs to")
st.image('pages/app_image.jpg')
# Model Selection
selected_model_name = st.selectbox(
    "Choose a model:", list(model_options.keys())
)

# User Input
st.header('Enter Text:')
user_text = st.text_area('Type or paste your text here', height=200)

# Prediction Button
if st.button('Classify'):
    if user_text:
        # Load the selected model
        selected_model_file = model_options[selected_model_name]
        with open(selected_model_file, 'rb') as file:
            model = pickle.load(file)

        try:
            # Make Prediction
            prediction = model.predict([user_text])[0]

            # Display Prediction
            st.subheader('Category:')
            st.write(prediction)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning('Please enter some text.')

# Add link button after your existing content
st.markdown("<h3 style='text-align: center;'>Learn More About Us</h3>", unsafe_allow_html=True)
st.link_button("About Us", "pages/about_us.py")