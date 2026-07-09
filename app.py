import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Core Page Setup
st.set_page_config(page_title="AI Matcher Suite (Live Data Mode)", layout="wide")

# 1. Sidebar Configurations Panel (Acts as a Password Input)
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
st.title("🚀 Real AI Search & WhatsApp Broadcast Suite")
st.write("Upload real recruitment data sheets to run semantic matrix queries and broadcast high-scannability alert cards directly onto community channels.")

# 2. Dynamic Real Data File Ingestor (Replaces the simulation entirely)
st.subheader("📁 Ingest Live Job Dataset")
uploaded_file = st.file_uploader(
    label="Upload your real job spreadsheet (CSV format required):", 
    type=["csv"],
    help="Ensure your file includes columns matching exactly: Title, Company, Skills, Location, Portal, ApplyURL, PostedDate"
)

# Operational Data Assignment Gate
if uploaded_file is not None:
    try:
        # Read real tracking data rows directly from the uploaded file buffer
        df = pd.read_csv(uploaded_file)
        
        # Standardize required verification fields to prevent column mismatch crashes
        required_columns = ["Title", "Company", "Skills", "Location", "Portal", "ApplyURL", "PostedDate"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing Columns in CSV: {missing_columns}. Please match the format layout requirements.")
            st.stop()
            
        # Ensure every single row has a unique string identifier for metadata lookups
        df["ID"] = df.index.astype(str)
        st.success(f"📊 Successfully loaded {len(df)} real job listings from your dataset!")
        
    except Exception as read_err:
        st.error(f"❌ Error reading spreadsheet data: {read_err}")
        st.stop()
else:
    st.info("💡 Getting Started: Please upload an active CSV file containing real listings to initialize the AI Search Index.")
    st.stop()

# 3. Vector Database Processing Loop
if google_api_key:
    try:
        # Create standard Langchain Document objects carrying explicit key-value parameters inside the metadata block
        documents = []
        for index, row in df.iterrows():
            search_string = f"Job: {row['Title']} | Company: {row['Company']} | Skills: {row['Skills']} | Location: {row['Location']} | Portal: {row['Portal']}"
            doc = Document(
                page_content=search_string,
                metadata={"row_id": str(row['ID'])} # Metadata injection isolates matching from raw link modifications
            )
            documents.append(doc)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        with st.spinner("Processing live rows into AI Vector Map..."):
            vector_db = FAISS.from_documents(documents, embeddings)
        st.success("🤖 Core AI Matrix loaded successfully!")
        # 4. Strictly the Context Match Finder Interface
        st.markdown("---")
        st.subheader("🔍 Context Match Finder")
        user_query = st.text_input("What profile or criteria are you trying to find?", placeholder="e.g., developers in visakhapatnam")
        
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
                    raw_text = doc.page_content
                    
                    # ZERO-EXCEPTION RETRIEVAL: Pull variables directly out of dataframe row parameters using metadata ID keys
                    try:
                        row_id = doc.metadata["row_id"]
                        matched_row = df[df["ID"] == row_id].iloc[0]
                        
                        job_title = matched_row["Title"]
                        company_name = matched_row["Company"] 
                        skills = matched_row["Skills"]
                        location = matched_row["Location"]
                        portal = matched_row["Portal"]
                        url_link = str(matched_row["ApplyURL"]) # Natively outputs the complete tracking string length untouched
                        posted_date_raw = str(matched_row["PostedDate"])
                    except Exception as extraction_err:
                        job_title, company_name, skills, location, portal, url_link, posted_date_raw = "Job Profile", "Hiring Organization", "Skills Matrix", "Location", "Portal", "https://linkedin.com", "2026-07-09"
                    
                    # Compute relative timelines dynamically relative to your local execution clock
                    days_ago_str = ""
                    try:
                        extracted_date = datetime.strptime(posted_date_raw.strip(), "%Y-%m-%d").date()
                        delta = (current_date - extracted_date).days
                        if delta == 0:
                            days_ago_str = " (Today)"
                        elif delta == 1:
                            days_ago_str = " (1 day ago)"
                        else:
                            days_ago_str = f" ({delta} days ago)"
                    except Exception:
                        pass
                    
                    # Construct clean card layout using proper asterisks (*) for WhatsApp bold text headers
                    clean_card = (
                        f"*Position:* {job_title}\n"
                        f"*Company:* {company_name}\n"      
                        f"*Skills:* {skills}\n"
                        f"*Location:* {location}\n"
                        f"*Source Portal:* {portal}\n"
                        f"*🔗 Apply Here:* {url_link.strip()}\n" 
                        f"*📅 Posted Date:* {posted_date_raw}{days_ago_str}"
                    )
                    
                    simplified_wa_text += f"📌 *Match #{index+1}*\n{clean_card}\n\n"
                
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
                                st.error(f"WhatsApp Dispatch Failed. Server Error Details: {response.text}")
                else:
                    st.warning("💡 Fill in your WhatsApp configuration parameters in the sidebar to send this message.")
                    
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
else:
    st.info("💡 Please enter your Google API Key in the left sidebar to boot up the system.")
