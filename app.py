import streamlit as st
import pandas as pd
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Set page title and layout
st.set_page_config(page_title="Dynamic AI Search Engine", layout="wide")

# 1. Sidebar Panel for Multi-Model Credentials & Options
st.sidebar.title("🔐 API Authentications")

# Core Embedding Key (Required for finding records)
google_api_key = st.sidebar.text_input(
    label="1. Google AI API Key (Required for Vector Mapping):",
    type="password",
    placeholder="Enter your google_api_key here...",
    help="Used to generate mathematical search vectors via gemini-embedding-001."
)
st.sidebar.markdown("[Get a Free Google API Key ↗️](https://google.com)")

st.sidebar.markdown("---")

# User Choice Selection Dropdown
st.sidebar.subheader("🤖 Reasoning Engine Selection")
ai_provider = st.sidebar.selectbox(
    "Choose your summary provider:",
    options=["Google Gemini", "Groq Llama 3"]
)

# Conditional input depending on dropdown selection
groq_api_key = ""
if ai_provider == "Groq Llama 3":
    groq_api_key = st.sidebar.text_input(
        label="2. Groq Cloud API Key:",
        type="password",
        placeholder="Enter your groq_api_key here...",
        help="Required to run Llama 3 reasoning acceleration."
    )
    st.sidebar.markdown("[Get a Free Groq API Key ↗️](https://groq.com)")

st.title("🚀 Custom Dual-Engine AI Search Suite")
st.write(f"This platform indexes matching data clusters using Google Embeddings and explains them via your chosen engine: **{ai_provider}**.")

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
        
        # Connect to Google's current text-embedding engine
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        # Transform words into mathematical vectors stored inside memory
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
            
            # Extract raw text from matches to feed into the chosen system prompt
            raw_context = "\n".join([doc.page_content for doc in matched_results])
            
            system_prompt = f"""
            You are an expert recruitment assistant. 
            Based strictly on the following database records:
            {raw_context}
            
            Synthesize an explanation for the search query: "{user_query}"
            Format the output with clean markdown text. Break down why these listings are relevant.
            """

            # ROUTE 1: Google Gemini selected
            if ai_provider == "Google Gemini":
                with st.spinner("✨ Google Gemini reasoning model calculating reply..."):
                    # Use Langchain's native Gemini call function
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        google_api_key=google_api_key,
                        temperature=0.3
                    )
                    response = llm.invoke(system_prompt)
                    ai_response = response.content
                
                st.markdown("### 🏆 Top Database Matches Explained (Google Gemini):")
                st.info(ai_response)

            # ROUTE 2: Groq Llama 3 selected
            elif ai_provider == "Groq Llama 3":
                if groq_api_key:
                    with st.spinner("⚡ Groq LPU computing Llama 3 response..."):
                        client = Groq(api_key=groq_api_key)
                        completion = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": system_prompt}],
                            temperature=0.3
                        )
                        ai_response = completion.choices.message.content
                    
                    st.markdown("### 🏆 Top Database Matches Explained (Groq Llama 3):")
                    st.info(ai_response)
                else:
                    st.error("🔑 Groq Key Missing: Please provide your Groq API Key in the left sidebar to generate the summary statement.")
                    
    except Exception as e:
        st.error(f"❌ Initialization Error: Check your key structure. Details: {e}")
else:
    st.info("💡 Application Paused: Please enter your Google API Key in the left sidebar configuration panel to test the AI search index.")
