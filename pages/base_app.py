import streamlit as st
import pickle
import os

# Model Options with Absolute Paths
model_options = {
    "Logistic Regression": os.path.join(os.path.dirname(__file__), 'models/logistic_regression_pipeline.pkl'),
    "Random Forest": os.path.join(os.path.dirname(__file__), 'models/random_forest_model.pkl'),
    "KNN Classifier": os.path.join(os.path.dirname(__file__), 'models/knn_pipeline.pkl')
}

# Set page config (title, icon) ONLY for the main page
st.set_page_config(
    page_title="Text Prediction App", page_icon=":crystal_ball:"
)

# App Title
st.title('Story Stream')
st.write("This app will take in some text data you have, and predict the category this belongs to")
st.image('app_image.jpg')

# Manual Sidebar Navigation
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "main_page"

if st.sidebar.button("Home"):
    st.session_state.page = "main_page"  
    st.experimental_rerun()  # Re-run the script to reflect the change
if st.sidebar.button("About Us"):
    st.session_state.page = "about_us"
    st.experimental_rerun()  
if st.sidebar.button("Explore Data"):
    st.session_state.page = "explore_data"
    st.experimental_rerun()  

if st.session_state.page == 'about_us':
    st.title("About Story Stream")

    st.markdown(
    """
    :black[Welcome to Story Stream! We are a team of passionate storytellers and data scientists 
    dedicated to helping you understand and categorize your text data.]

    **Our Mission:**
    :black[To empower individuals and organizations to unlock the insights hidden within their text data.]

    **Our Team:**
    
    - **Neo Modibedi:** Experienced software engineer specializing in building scalable web applications.
    - **Thapelo Robyn Raphala:** Lead Data Scientist with 10+ years of experience in NLP and machine learning.
    - **Thandekile Sikhakhane:** Creative writer and content strategist passionate about storytelling with data.
    - **Mbalenhle Lenepa:** Data Scientist with 10+ years of experience in NLP and machine learning.


    **Contact Us:**
    
    - **Email:** storystream@example.com
    - **Website:** https://www.storystream.com 
    

    We'd love to hear from you! Try out our text prediction app on the "Predict" page and let us know what you think.
    """
    )

elif st.session_state.page == 'explore_data':
    # Code for your "Explore Data" page
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    st.title("Explore Your Data")

    # User Input (Text Area)
    st.header("Enter Text:")
    text = st.text_area("Type or paste your text here", height=200)

    # Word Cloud Generation
    if text:
        # Combine default stopwords with additional custom stopwords
        stopwords = set(STOPWORDS)
        additional_stopwords = {'said', 'would', 'could', 'also'}  # Add more as needed
        stopwords.update(additional_stopwords)

        # Generate Word Cloud
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

        # Display Word Cloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)

    else:
        st.warning("Please enter some text to generate a word cloud.")
else: #This is your original main page
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
            with st.spinner('Classifying...'):
                try:
                    selected_model_file = model_options[selected_model_name]
                    if not os.path.exists(selected_model_file):
                        raise FileNotFoundError(f"Model file not found: {selected_model_file}")  
                    
                    with open(selected_model_file, 'rb') as file:
                        model = pickle.load(file)

                    prediction = model.predict([user_text])[0]
                    st.success(f"Text classified successfully as '{prediction}!'")

                except FileNotFoundError as fnf_error:
                    st.error(f"Model not found. Please check the model path: {fnf_error}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning('Please enter some text.')

    # Model Descriptions
    with st.expander("About the Models"):
        st.write("**Logistic Regression:** A simple yet effective linear model for classification.")
        st.write("**Random Forest:**  An ensemble model that combines multiple decision trees for improved accuracy.")
        st.write("**Support Vector Machine:** A powerful model for finding complex patterns in data.")
