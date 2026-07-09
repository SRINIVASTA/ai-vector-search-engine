import streamlit as st
import pandas as pd
import requests
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Core Page Setup
st.set_page_config(page_title="AI Matcher & Broadcast Suite", layout="wide")

# 1. Sidebar Configurations Panel
st.sidebar.title("🔐 API Authentications")

# Core Google Key Input
google_api_key = st.sidebar.text_input(
    label="1. Google AI API Key (Required for Vectors):",
    type="password",
    placeholder="Enter your google_api_key here...",
    help="Required to index database entries via gemini-embedding-001."
)
st.sidebar.markdown("[Get Google API Key ↗️](https://aistudio.google.com/)")

st.sidebar.markdown("---")

# Reasoning Engine Selector Switch
st.sidebar.subheader("🤖 Reasoning Engine Selection")
ai_provider = st.sidebar.selectbox(
    "Choose your summary provider:",
    options=["Google Gemini", "Groq Llama 3"]
)

# Conditional display for Groq Key
groq_api_key = ""
if ai_provider == "Groq Llama 3":
    groq_api_key = st.sidebar.text_input(
        label="2. Groq Cloud API Key:",
        type="password",
        placeholder="Enter your groq_api_key here...",
        help="Required to run high-speed Llama 3 summarization."
    )
    st.sidebar.markdown("[Get a Free Groq API Key ↗️](https://groq.com)")

st.sidebar.markdown("---")

# WhatsApp Broadcast Parameters Panel
st.sidebar.subheader("📱 WhatsApp Community Settings")
wa_instance = st.sidebar.text_input("WhatsApp Instance ID:", placeholder="e.g., 11011234")
wa_token = st.sidebar.text_input("WhatsApp Gateway Token:", type="password", placeholder="Enter token...")
wa_chat_id = st.sidebar.text_input("Target Group/Community ID:", placeholder="e.g., 1203632@g.us")

# Main Canvas Header
st.title("🚀 Two-Stage AI Search & WhatsApp Broadcast Suite")
st.write(f"Index data records via Google Vector embeddings, summarize them using **{ai_provider}**, and blast notifications directly to WhatsApp.")

# 2. Local Database Mock Dataset
@st.cache_data
def load_mock_data():
    return pd.DataFrame({
        "Title": ["Frontend React Developer", "Data Scientist", "DevOps Engineer", "HR Manager"],
        "Skills": ["React, JavaScript, CSS", "Python, Machine Learning, SQL", "Docker, AWS, Linux", "Hiring, Payroll, Excel"],
        "Location": ["Remote", "Mumbai", "Bangalore", "Delhi"]
    })

df = load_mock_data()
st.subheader("📊 Current Active Database Listings")
st.dataframe(df, use_container_width=True)

# 3. Vector Database Processing Loop
if google_api_key:
    try:
        # Preprocess data fields for vector layout ingestion
        df["search_text"] = "Job: " + df["Title"] + " | Skills: " + df["Skills"] + " | Location: " + df["Location"]
        text_records = df["search_text"].tolist()
        
        # Instantiate Google Embedding Engine (Fixed working production string)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        with st.spinner("Processing rows into AI Vector Map..."):
            vector_db = FAISS.from_texts(text_records, embeddings)
        st.success("✅ AI Embeddings generated and indexed successfully!")

        # 4. Search and Synthesis Block
        st.markdown("---")
        st.subheader("🔍 Context Match Finder")
        user_query = st.text_input("What profile or criteria are you trying to find?")
        
        if user_query:
            with st.spinner("Scanning vector clusters for closest matches..."):
                matched_results = vector_db.similarity_search(user_query, k=2)
            
            raw_context = "\n".join([doc.page_content for doc in matched_results])
            
            # Formulate foundational instruction prompt
            system_prompt = f"""
            You are an expert recruitment assistant. 
            Based strictly on the following database records:
            {raw_context}
            
            Synthesize an explanation for the search query: "{user_query}"
            Format the output with clean markdown text. Break down why these listings are relevant.
            """
            
            ai_response = ""

            # Path A: Gemini Processing 
            if ai_provider == "Google Gemini":
                with st.spinner("✨ Google Gemini model calculating reply..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",  # Direct up-to-date API string mapping
                        google_api_key=google_api_key,
                        temperature=0.3
                    )
                    response = llm.invoke(system_prompt)
                    ai_response = response.content
                st.markdown("### 🏆 Top Database Matches Explained (Google Gemini):")
                st.info(ai_response)

            # Path B: Groq Llama 3 Processing
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

            # 5. Active WhatsApp Broadcast Engine Interface Trigger
            if ai_response:
                st.markdown("---")
                st.subheader("📤 Community Broadcast Panel")
                
                if wa_instance and wa_token and wa_chat_id:
                    if st.button("🚀 Push Summary to WhatsApp Community", type="primary"):
                        with st.spinner("Transmitting data packet to WhatsApp API..."):
                            # HTTP Endpoint path setup
                            url = f"https://green-api.com{wa_instance}/sendMessage/{wa_token}"
                            
                            # Build text payload converting markdown components to WhatsApp format
                            whatsapp_message = f"📢 *AI Match Analysis Alert*\n\n{ai_response}\n\n_Sent via Streamlit AI Engine_"
                            
                            payload = {"chatId": wa_chat_id, "message": whatsapp_message}
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, json=payload, headers=headers)
                            
                            if response.status_code == 200:
                                st.success("🎉 Successfully posted to your WhatsApp Community chat dashboard!")
                            else:
                                st.error(f"WhatsApp Dispatch Failed. API Response Error: {response.text}")
                else:
                    st.warning("💡 To broadcast these notes, fill in the WhatsApp panel properties located on the left menu sidebar.")
                    
    except Exception as e:
        st.error(f"❌ Application Error encountered: {e}")
else:
    st.info("💡 Application Paused: Please enter your Google API Key in the left sidebar configuration panel to boot up the vector engine.")
