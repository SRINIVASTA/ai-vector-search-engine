import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Core Page Setup
st.set_page_config(page_title="AI Matcher Suite", layout="wide")

# 1. Sidebar Configurations Panel
st.sidebar.title("🔐 API Authentications")

google_api_key = st.sidebar.text_input(
    label="1. Google AI API Key:",
    type="password",
    placeholder="Enter your google_api_key here...",
    help="Required to index database entries."
)
st.sidebar.markdown("[Get Google API Key ↗️](https://google.com)")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Reasoning Engine Selection")
ai_provider = st.sidebar.selectbox(
    "Choose your summary provider:",
    options=["Local Formatter (Zero Quota Limit)", "Groq Llama 3 (Fast LLM)"]
)

groq_api_key = ""
if ai_provider == "Groq Llama 3 (Fast LLM)":
    groq_api_key = st.sidebar.text_input(
        label="2. Groq Cloud API Key:",
        type="password",
        placeholder="Enter your groq_api_key here..."
    )
    st.sidebar.markdown("[Get a Free Groq API Key ↗️](https://groq.com)")

st.sidebar.markdown("---")

st.sidebar.subheader("📱 WhatsApp Community Settings")
wa_instance = st.sidebar.text_input("WhatsApp Instance ID:", placeholder="e.g., 11011234")
wa_token = st.sidebar.text_input("WhatsApp Gateway Token:", type="password", placeholder="Enter token...")
wa_chat_id = st.sidebar.text_input("Target Group/Community ID:", placeholder="e.g., 1203632@g.us")

# Main Canvas Header
st.title("🚀 AI Search & WhatsApp Broadcast Suite")
st.write("An embedding-powered matching layout designed to parse custom profiles and push updates directly to messaging networks.")

# 2. Mock Dataset (Upgraded with Timestamps)
@st.cache_data
def load_mock_data():
    return pd.DataFrame({
        "Title": ["Frontend React Developer", "Data Scientist", "DevOps Engineer", "HR Manager"],
        "Skills": ["React, JavaScript, CSS", "Python, Machine Learning, SQL", "Docker, AWS, Linux", "Hiring, Payroll, Excel"],
        "Location": ["Remote", "Mumbai", "Bangalore", "Delhi"],
        "Portal": ["Naukri.com", "LinkedIn", "Indeed", "Glassdoor"],
        "ApplyURL": [
            "https://naukri.com",
            "https://linkedin.com",
            "https://indeed.com",
            "https://glassdoor.com"
        ],
        "PostedDate": ["2026-07-08", "2026-07-07", "2026-07-05", "2026-06-25"]  # YYYY-MM-DD dynamic formats
    })

df = load_mock_data()

# Sliding Example Expander (Reflecting structural timeline metrics)
with st.expander("💡 Click to view example database format reference"):
    st.write("Your backend or input data evaluates records matching this structural alignment:")
    st.dataframe(df, use_container_width=True)

# 3. Vector Database Processing Loop
if google_api_key:
    try:
        # Preprocess and merge all dimensions including Date indices
        df["search_text"] = (
            "Job: " + df["Title"] + 
            " | Skills: " + df["Skills"] + 
            " | Location: " + df["Location"] + 
            " | Portal: " + df["Portal"] + 
            " | ApplyURL: " + df["ApplyURL"] +
            " | PostedDate: " + df["PostedDate"]
        )
        text_records = df["search_text"].tolist()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        with st.spinner("Processing rows into AI Vector Map..."):
            vector_db = FAISS.from_texts(text_records, embeddings)
        st.success("✅ AI Embeddings indexed successfully!")

        # 4. Search and Synthesis Block
        st.markdown("---")
        st.subheader("🔍 Context Match Finder")
        user_query = st.text_input("What profile or criteria are you trying to find?", placeholder="e.g., data scientist, Mumbai")
        
        if user_query:
            with st.spinner("Scanning vector clusters for closest matches..."):
                matched_results = vector_db.similarity_search(user_query, k=2)
            
            ai_response_check = False
            groq_response_text = ""

            if ai_provider == "Local Formatter (Zero Quota Limit)":
                ai_response_check = True

            elif ai_provider == "Groq Llama 3 (Fast LLM)":
                if groq_api_key:
                    with st.spinner("⚡ Processing query via Groq LPU..."):
                        raw_context = "\n".join([doc.page_content for doc in matched_results])
                        system_prompt = f"Context:\n{raw_context}\nQuery: {user_query}. Summarize nicely."
                        
                        client = Groq(api_key=groq_api_key)
                        completion = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{"role": "user", "content": system_prompt}],
                            temperature=0.3
                        )
                        groq_response_text = completion.choices.message.content
                        ai_response_check = True
                else:
                    st.error("🔑 Groq Key Missing. Please provide it in the sidebar.")

            # 5. Build and Display Clean Output Card
            if ai_response_check:
                st.markdown("### 🏆 Top Database Matches Found:")
                
                simplified_wa_text = f"📢 *New Job Match Alert!*\n\n*Query:* {user_query}\n\n"
                
                if groq_response_text:
                    simplified_wa_text += f"*AI Summary:*\n{groq_response_text}\n\n---\n\n"
                
                current_date = datetime.now().date()
                
                for index, doc in enumerate(matched_results):
                    # Extract the raw content components
                    raw_text = doc.page_content
                    
                    # Calculate human-friendly "days ago" metadata on the fly
                    days_ago_str = ""
                    try:
                        date_part = raw_text.split(" | PostedDate: ")[1].strip()
                        extracted_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                        delta = (current_date - extracted_date).days
                        if delta == 0:
                            days_ago_str = " (Today)"
                        elif delta == 1:
                            days_ago_str = " (1 day ago)"
                        else:
                            days_ago_str = f" ({delta} days ago)"
                    except Exception:
                        pass
                    
                    # Custom cleanup replacement rules
                    clean_item = (
                        raw_text
                        .replace("Job: ", "*Position:* ")
                        .replace(" | Skills: ", "\n*Skills:* ")
                        .replace(" | Location: ", "\n*Location:* ")
                        .replace(" | Portal: ", "\n*Source Portal:* ")
                        .replace(" | ApplyURL: ", "\n*🔗 Apply Here:* ")
                        .replace(" | PostedDate: ", "\n*📅 Posted Date:* ")
                    )
                    
                    # Append relative human-friendly date data onto card blocks
                    clean_item += days_ago_str
                    
                    simplified_wa_text += f"📌 *Match #{index+1}*\n{clean_item}\n\n"
                
                simplified_wa_text += "🤖 _Sent via AI Matcher Suite_"

                # Render clean mobile UI card on the dashboard
                st.info(simplified_wa_text)
                
                # 6. Active WhatsApp Broadcast Engine Logic
                if wa_instance and wa_token and wa_chat_id:
                    if st.button("🚀 Push Message to WhatsApp Community", type="primary"):
                        with st.spinner("Transmitting to WhatsApp..."):
                            url = f"https://green-api.com{wa_instance}/sendMessage/{wa_token}"
                            payload = {"chatId": wa_chat_id, "message": simplified_wa_text}
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, json=payload, headers=headers)
                            if response.status_code == 200:
                                st.success("🎉 Message posted to your WhatsApp Community successfully!")
                            else:
                                st.error(f"WhatsApp Dispatch Failed: {response.text}")
                else:
                    st.warning("💡 Fill in your WhatsApp configuration parameters in the sidebar to send this message.")
                    
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
else:
    st.info("💡 Please enter your Google API Key in the left sidebar to boot up the system.")
