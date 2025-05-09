import streamlit as st 
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import time
from urllib.parse import urlparse
import tweepy
import praw
import numpy as np
import logging
import json  # For Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ========== HEADER ==========
st.markdown("""
    <div style='background-color:#012169;padding:10px;text-align:center;'>
        <img src="https://www.bcu.ac.uk/images/default-source/marketing/logos/bcu-logo.svg" width="200"/>
    </div>
""", unsafe_allow_html=True)

def footer():
    st.markdown("""
    <hr>
    <div style='text-align:center; color: gray;'>
        Developed by <strong>Doaa Al-Turkey</strong>
    </div>
    """, unsafe_allow_html=True)

# ========== SECRETS ==========
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
    REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
    REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
    TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]
    GOOGLE_SHEETS_CREDENTIALS_JSON = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    GOOGLE_SHEET_NAME = st.secrets["GOOGLE_SHEET_NAME"]

    google_creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
    scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    google_credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scopes)
    gc = gspread.authorize(google_credentials)
    feedback_sheet = gc.open(GOOGLE_SHEET_NAME).sheet1

except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()
except Exception as e:
    st.error(f"Google Sheets init failed: {e}")
    st.stop()

# ========== INIT ==========
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"Error initializing APIs: {e}")
    st.stop()
# ========== BNF INFO ==========
def fetch_bnf_info(medicine_name):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    base_url = "https://bnf.nice.org.uk"
    search_url = f"{base_url}/search/?q={medicine_name.replace(' ', '%20')}"
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = None
    content = ""

    try:
        driver = webdriver.Chrome(options=options)
        driver.get(search_url)
        WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "card__header")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = [base_url + a['href'] for a in soup.select("header.card__header a[href^='/']")[:3]]

        for link in links:
            driver.get(link)
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            page_soup = BeautifulSoup(driver.page_source, "html.parser")
            content += page_soup.get_text(separator="\n", strip=True) + "\n"

    except Exception as e:
        st.error(f"BNF scraping error: {e}")
    finally:
        if driver:
            driver.quit()

    return content

# ========== NHS INFO ==========
def fetch_nhs_info(query):
    base_url = "https://www.nhs.uk"
    search_url = f"{base_url}/search/"
    params = {"q": query}
    content = ""

    try:
        session = requests.Session()
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = session.get(search_url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a['href'] for a in soup.select('a[href*="/conditions/"], a[href*="/medicines/"]')[:5]]

        for href in links:
            page_url = href if href.startswith("http") else base_url + href
            resp = session.get(page_url)
            page_soup = BeautifulSoup(resp.text, "html.parser")
            main = page_soup.find("main") or page_soup.find("article")
            if main:
                content += main.get_text(separator="\n", strip=True) + "\n"

    except Exception as e:
        st.error(f"NHS fetching error: {e}")

    return content

# ========== WEBSITE FETCH ==========
def fetch_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        main = soup.find('main') or soup.find('article') or soup.body
        return main.get_text(separator="\n", strip=True) if main else ""
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

# ========== GPT RESPONSE ==========
def generate_response(user_input, reference_text):
    prompt = (
        f"You are a helpful medical assistant. Based only on the reference info below, summarize or respond to:\n\n"
        f"{user_input}\n\nReference:\n{reference_text[:12000]}"
    )
    messages = [
        {"role": "system", "content": "You respond using only the reference text. No external facts."},
        {"role": "user", "content": prompt}
    ]
    try:
        res = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.4, max_tokens=1000
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ GPT Error: {e}"

# ========== SIMILARITY ==========
def compute_similarity_score(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    try:
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b).item()
        return round(sim * 100, 2)
    except:
        return 0.0

# ========== FACT-CHECK ==========
def fact_check_post(post_text, reference_text):
    prompt = (
        "You're a medical fact-checker. Compare each claim in this post to the verified medical reference.\n"
        "✅ if supported, ❌ if contradicted, ⚠️ if unverifiable.\n\n"
        f"POST:\n\"\"\"{post_text}\"\"\"\n\nREFERENCE:\n\"\"\"{reference_text[:8000]}\"\"\"\n\nBegin analysis:"
    )
    messages = [
        {"role": "system", "content": "You highlight verified vs unverified medical claims."},
        {"role": "user", "content": prompt}
    ]
    try:
        res = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3, max_tokens=800
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Fact-check error: {e}"

# ========== SAVE FEEDBACK ==========
def save_feedback(record):
    try:
        feedback_sheet.append_row([
            record.get("Timestamp", ""),
            record.get("Input", ""),
            record.get("Type", ""),
            record.get("Source", ""),
            record.get("Summary", ""),
            record.get("Feedback", ""),
            record.get("Similarity Score (Summary vs Source)", "")
        ])
        st.success("✅ Feedback saved.")
    except Exception as e:
        st.error(f"Google Sheets error: {e}")
# ========== MAIN UI ==========
st.title("💊 Validate Medicine Information from Social Media & Web")

input_type = st.radio("What would you like to analyze?", 
                      ["Medicine", "Medical Query", "Webpage with Medical Claims"], 
                      horizontal=True)

user_input_label = {
    "Medicine": "Enter a medicine name (e.g., paracetamol):",
    "Medical Query": "Enter a medical question (e.g., causes of high blood pressure):",
    "Webpage with Medical Claims": "Enter a URL of a webpage with medical claims:"
}[input_type]

user_input = st.text_input(user_input_label)

# Session State
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'reference_text' not in st.session_state: st.session_state.reference_text = ""
if 'generated_summary' not in st.session_state: st.session_state.generated_summary = ""
if 'source_name' not in st.session_state: st.session_state.source_name = ""

if st.button("Analyze"):
    st.session_state.analysis_done = False
    st.session_state.reference_text = ""
    st.session_state.generated_summary = ""
    st.session_state.source_name = ""

    if not user_input.strip():
        st.warning("❗ Please enter valid input.")
    else:
        with st.spinner("🔍 Fetching reference data..."):
            reference = ""
            source = ""

            if input_type == "Medicine":
                reference = fetch_bnf_info(user_input)
                source = "BNF"
            elif input_type == "Medical Query":
                reference = fetch_nhs_info(user_input)
                source = "NHS"
            elif input_type == "Webpage with Medical Claims":
                if not user_input.startswith("http"):
                    st.error("❌ Please enter a valid URL (http:// or https://)")
                else:
                    reference = fetch_url_content(user_input)
                    source = "Webpage"

            if reference.strip():
                st.session_state.reference_text = reference
                st.session_state.source_name = source
                st.session_state.generated_summary = generate_response(user_input, reference)
                st.session_state.analysis_done = True
            else:
                st.error(f"❌ Could not retrieve reference info from {source}.")

# ========== RESULTS ==========
if st.session_state.analysis_done:
    st.markdown(f"### 🧠 GPT Summary (Based on {st.session_state.source_name} Info):")
    st.write(st.session_state.generated_summary)

    if st.session_state.source_name in ["NHS", "Webpage"]:
        score = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
        st.markdown(f"**🔗 Similarity Score:** {score}/100")
        st.caption("This indicates how closely the GPT summary matches the source.")

    if st.session_state.source_name == "Webpage":
        st.markdown("### 🧾 Claim Verification from Webpage")
        with st.spinner("🔍 Fact-checking webpage content..."):
            check = fact_check_post(st.session_state.reference_text, st.session_state.reference_text)
            st.write(check)

# ========== FEEDBACK ==========
    with st.expander("📝 Doctor Feedback"):
        feedback_text = st.text_area("Please provide your feedback on the summary:")
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                record = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input": user_input,
                    "Type": input_type,
                    "Source": st.session_state.source_name,
                    "Summary": st.session_state.generated_summary,
                    "Feedback": feedback_text
                }
                if st.session_state.source_name in ["NHS", "Webpage"]:
                    record["Similarity Score (Summary vs Source)"] = score
                save_feedback(record)
            else:
                st.warning("Please enter feedback before submitting.")

# ========== FACT-CHECKING SOCIAL POSTS ==========
    if st.session_state.source_name in ["BNF", "NHS"]:
        st.markdown("## 🔍 Fact-Check Social Media Posts")

        tabs = st.tabs(["🐦 Twitter", "💬 Reddit"])

        with tabs[0]:
            st.markdown(f"### 🐦 Tweets mentioning '{user_input}'")
            try:
                tweets = twitter_client.search_recent_tweets(query=f"{user_input} -is:retweet lang:en", 
                                                             max_results=5, tweet_fields=["text"])
                for i, tweet in enumerate(tweets.data if tweets.data else []):
                    text = tweet.text
                    st.markdown(f"**Tweet {i+1}:**")
                    st.info(text)
                    with st.spinner("Fact-checking tweet..."):
                        fc = fact_check_post(text, st.session_state.reference_text)
                        cred = compute_similarity_score(text, st.session_state.reference_text)
                        st.write(fc)
                        st.markdown(f"🧮 Similarity Score: {cred}/100")
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Twitter Error: {e}")

        with tabs[1]:
            st.markdown(f"### 💬 Reddit Posts mentioning '{user_input}'")
            try:
                posts = reddit.subreddit("medicine").search(user_input, sort="new", limit=5)
                for i, post in enumerate(posts):
                    text = post.title + "\n\n" + post.selftext
                    st.markdown(f"**Reddit Post {i+1}:** {post.title}")
                    with st.expander("View & Fact-Check"):
                        st.caption(post.selftext)
                        with st.spinner("Fact-checking..."):
                            fc = fact_check_post(text, st.session_state.reference_text)
                            cred = compute_similarity_score(text, st.session_state.reference_text)
                            st.write(fc)
                            st.markdown(f"🧮 Similarity Score: {cred}/100")
            except Exception as e:
                st.warning(f"Reddit Error: {e}")

# ========== FOOTER ==========
footer()
