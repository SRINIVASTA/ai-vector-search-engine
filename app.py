import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Core Page Setup
st.set_page_config(page_title="AI Matcher Suite", layout="wide")

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
# 2. Hardcoded Database Loader (Perfectly balanced column rows)
@st.cache_data
def load_mock_data():
    today = datetime.now().date()
    
    # Generate relative timelines automatically
    date_today = today.strftime("%Y-%m-%d")
    date_yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    date_2_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    date_4_days_ago = (today - timedelta(days=4)).strftime("%Y-%m-%d")
    
    return pd.DataFrame({
        "ID": ["0", "1", "2", "3", "4", "5", "6"],
        "Title": [
            "Frontend React Developer", 
            "Data Scientist",               
            "DevOps Engineer", 
            "HR Manager",
            "Python Backend Engineer",       
            "Full Stack Node Developer",
            "Junior Data Scientist"          
        ],
        "Company": [
            "Moprens Solutions",          
            "TechCorp Global",
            "CloudScale Tech",
            "Global Nexus HR",
            "Vizag Innovate Labs",        
            "Sankhya Technologies",       
            "Alpha Analytics India"       
        ],
        "Skills": [
            "React, JavaScript, CSS", 
            "Python, Machine Learning, SQL", 
            "Docker, AWS, Linux", 
            "Hiring, Payroll, Excel",
            "Python, Django, PostgreSQL, REST APIs", 
            "Node.js, Express, React, MongoDB",
            "Python, SQL, Predictive Analytics, Tableau" 
        ],
        "Location": [
            "Remote", 
            "Mumbai", 
            "Bangalore", 
            "Delhi",
            "Visakhapatnam",                
            "Visakhapatnam",
            "Visakhapatnam"                 
        ],
        "Portal": [
            "Naukri.com", 
            "LinkedIn", 
            "Indeed", 
            "Glassdoor", 
            "Naukri.com", 
            "LinkedIn",
            "Naukri.com"
        ],
        "ApplyURL": [
            "https://naukri.com",
            "https://linkedin.com",
            "https://indeed.com",
            "https://glassdoor.co.in",
            "https://naukri.com",
            "https://linkedin.com",
            "https://naukri.com"
        ],
        "PostedDate": [
            date_yesterday,   
            date_yesterday,   
            date_2_days_ago,  
            date_4_days_ago,  
            date_today,       
            date_today,        
            date_today                       
        ]
    })

df = load_mock_data()

# 3. Vector Database Processing Loop
if google_api_key:
    try:
        df["search_text"] = (
            "ID: " + df["ID"] +
            " | Job: " + df["Title"] + 
            " | Company: " + df["Company"] + 
            " | Skills: " + df["Skills"] + 
            " | Location: " + df["Location"] + 
            " | Portal: " + df["Portal"]
        )
        text_records = df["search_text"].tolist()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=google_api_key
        )
        
        with st.spinner("Processing rows into AI Vector Map..."):
            vector_db = FAISS.from_texts(text_records, embeddings)
        st.success("✅ AI Embeddings indexed successfully!")
        # 4. Strictly the Context Match Finder Interface
        st.markdown("---")
        st.subheader("🔍 Context Match Finder")
        user_query = st.text_input("What profile or criteria are you trying to find?", placeholder="e.g., developer, visakhapatnam")
        
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
                    
                    try:
                        row_id = raw_text.split("ID: ")[1].split(" | Job: ")[0].strip()
                        matched_row = df[df["ID"] == row_id].iloc[0]
                        
                        job_title = matched_row["Title"]
                        company_name = matched_row["Company"] 
                        skills = matched_row["Skills"]
                        location = matched_row["Location"]
                        portal = matched_row["Portal"]
                        url_link = matched_row["ApplyURL"] 
                        posted_date_raw = matched_row["PostedDate"]
                    except Exception:
                        job_title, company_name, skills, location, portal, url_link, posted_date_raw = "Job Profile", "Hiring Organization", "Skills Matrix", "Location", "Portal", "https://linkedin.com", "2026-07-09"
                    
                    # Compute relative timelines
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
                    
                    # Construct clean card layouts with bold anchors
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
                            # 🚀 FILED API ROUTE: Secure network endpoint structure
                            url = f"https://green-api.com{wa_instance}/sendMessage/{wa_token}"
                            
                            payload = {"chatId": wa_chat_id, "message": simplified_wa_text}
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, json=payload, headers=headers)
                            if response.status_code == 200:
                                st.success("🎉 Message posted to your WhatsApp Community successfully!")
                            else:
                                st.error(f"WhatsApp Dispatch Failed. Server Error: {response.text}")
                else:
                    st.warning("💡 Fill in your WhatsApp configuration parameters in the sidebar to send this message.")
                    
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
else:
    st.info("💡 Please enter your Google API Key in the left sidebar to boot up the system.")
