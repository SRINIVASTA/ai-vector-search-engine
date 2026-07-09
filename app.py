import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set page title and layout
st.set_page_config(page_title="AI Search Engine (Google Mode)", layout="wide")

# 1. Sidebar Panel for Google Authentication
st.sidebar.title("🔐 Google Authentication")
google_api_key = st.sidebar.text_input(
    label="Enter your Google AI Studio API Key:",
    type="password",
    placeholder="Enter your google_api_key here...",
    help="Grab a key from Google AI Studio. It is processed in memory and never saved."
)

st.sidebar.markdown(
    "[Get a Free Google API Key ↗️](https://aistudio.google.com/)", 
    unsafe_allow_html=True
)

st.title("🎯 Vector Search Engine (Powered by Google AI)")
st.write("This app uses Google Embeddings to instantly scan your local database and find semantic matches.")

# 2. Local Database Mock Dataset
@st.cache_data
def load_mock_data():
    data = {
        "Title": ["Frontend React Developer", "Data Scientist", "DevOps Engineer", "HR Manager"],
        "Skills": ["React, JavaScript, CSS", "Python, Machine Learning, SQL", "Docker, AWS, Linux", "Hiring, Payroll, Excel"],
        "Location": ["Remote", "Mumbai", "Bangalore", "Delhi"]
    }
    return pd.DataFrame(data)

df = load_mock_data()

# Render database table
st.subheader("📊 Current Active Database Listings")
st.dataframe(df, use_container_width=True)

# 3. Operational Logic Loop
if google_api_key:
    try:
        # Preprocess and merge columns so the AI text encoder can index them
        df["search_text"] = "Job: " + df["Title"] + " | Skills: " + df["Skills"] + " | Location: " + df["Location"]
        text_records = df["search_text"].tolist()
        
        # Connect to Google's current text-embedding engine (Replaced retired model-001 string)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        # Transform the words into mathematical vectors stored inside memory
        with st.spinner("Processing rows into AI Vector Map..."):
            vector_db = FAISS.from_texts(text_records, embeddings)
        st.success("✅ AI Embeddings generated and indexed successfully!")

        # 4. Search Form Panel
        st.markdown("---")
        st.subheader("🔍 Context Match Finder")
        user_query = st.text_input("What profile or criteria are you trying to find?")
        
        if user_query:
            with st.spinner("Scanning vector clusters for closest matches..."):
                # Query FAISS database to find the top 2 closest database matches
                matched_results = vector_db.similarity_search(user_query, k=2)
            
            st.markdown("### 🏆 Top Database Matches Found:")
            for index, document in enumerate(matched_results):
                st.info(f"**Rank #{index+1}**: {document.page_content}")
                
    except Exception as e:
        st.error(f"❌ Initialization Error: Check your key structure. Details: {e}")
else:
    # Status alert warning displayed when sidebar is left empty
    st.info("💡 Application Paused: Please enter your Google API Key in the left sidebar configuration panel to test the AI search index.")
