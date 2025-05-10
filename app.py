# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import time
from urllib.parse import urlparse, quote
import tweepy
import praw
import numpy as np
import logging
import json
import gspread
from google.oauth2.service_account import Credentials

# ─── CONFIGURE LOGGING ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── STREAMLIT PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Medicine Info Validator",
    page_icon="🧠",
    layout="wide"
)

# ─── HEADER & FOOTER ──────────────────────────────────────────────
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

# ─── LOAD SECRETS ─────────────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
    REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
    REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
    TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]
    GOOGLE_SHEETS_CREDENTIALS_JSON = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    GOOGLE_SHEET_NAME = st.secrets["GOOGLE_SHEET_NAME"]

    # Parse Google Service Account JSON
    google_creds = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
    scopes = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(google_creds, scopes=scopes)
    gc = gspread.authorize(credentials)
    feedback_sheet = gc.open(GOOGLE_SHEET_NAME).sheet1

except KeyError as e:
    st.error(f"🔴 Missing secret: {e}")
    st.stop()
except Exception as e:
    st.error(f"🔴 Error initializing Google Sheets: {e}")
    st.stop()

# ─── INITIALIZE API CLIENTS ────────────────────────────────────────
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"🔴 Error initializing API clients: {e}")
    st.stop()

# ─── BNF SCRAPING (Requests-based) ─────────────────────────────────
def fetch_bnf_info(medicine_name: str, max_links: int = 5):
    """
    Fetches BNF search page summary, top result links, and detail-page text
    without Selenium (pure requests).
    """
    base_url = "https://bnf.nice.org.uk"
    search_url = f"{base_url}/search/?q={quote(medicine_name)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
    }

    results = {"search_summary": "", "links": [], "full_text": ""}

    # 1) GET search page
    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"BNF search request failed: {e}")
        return results

    soup = BeautifulSoup(resp.text, "html.parser")

    # 2) Extract summary from <main>
    main = soup.find("main", id="maincontent") or soup.find("main")
    if main:
        lines = [ln.strip() for ln in main.get_text("\n").splitlines() if ln.strip()]
        results["search_summary"] = "\n".join(lines[:20])

    # 3) Top links
    cards = soup.select("header.card__header a[href]")[:max_links]
    for a in cards:
        href = a["href"]
        title = a.get_text(strip=True)
        url = href if href.startswith("http") else base_url + href
        results["links"].append({"title": title, "url": url})

    # 4) Fetch each detail page
    for link in results["links"]:
        try:
            time.sleep(0.5)
            p = requests.get(link["url"], headers=headers, timeout=10)
            p.raise_for_status()
            ps = BeautifulSoup(p.text, "html.parser")
            topic = ps.find(id="topic") or ps.find("main") or ps.body
            text = topic.get_text("\n", strip=True) if topic else ps.get_text("\n", strip=True)
            results["full_text"] += f"## {link['title']}\n\n{text}\n\n"
        except Exception as e:
            logging.warning(f"BNF detail fetch failed {link['url']}: {e}")

    return results

# ─── NHS SCRAPING ──────────────────────────────────────────────────
def fetch_nhs_info(query: str, max_links: int = 5):
    base_url = "https://www.nhs.uk"
    search_url = f"{base_url}/search/"
    params = {"q": query}
    headers = {"User-Agent":"Mozilla/5.0"}

    content = ""
    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = soup.select('a[href*="/conditions/"], a[href*="/medicines/"]')[:max_links]
        for a in links:
            href = a["href"]
            page_url = href if href.startswith("http") else base_url + href
            p = requests.get(page_url, headers=headers, timeout=10)
            ps = BeautifulSoup(p.text, "html.parser")
            article = ps.find("article") or ps.find("main") or ps.body
            if article:
                content += article.get_text("\n", strip=True) + "\n\n"
    except Exception as e:
        logging.error(f"NHS fetch error: {e}")

    return content

# ─── GENERIC URL FETCH ───────────────────────────────────────────
def fetch_url_content(url: str):
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("article") or soup.find("main") or soup.body
        return main.get_text("\n", strip=True) if main else ""
    except Exception as e:
        logging.error(f"URL fetch error: {e}")
        return ""

# ─── GPT-4o RESPONSE ─────────────────────────────────────────────
def generate_response(user_input: str, reference_text: str) -> str:
    prompt = (
        "You are a medical assistant. Based *only* on the reference text below,\n"
        f"summarize or answer:\n\n{user_input}\n\nREFERENCE:\n{reference_text[:12000]}"
    )
    messages = [
        {"role":"system","content":"Use only the reference. Do not invent facts."},
        {"role":"user","content":prompt}
    ]
    try:
        res = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3, max_tokens=1000
        )
        return res.choices[0].message.content
    except Exception as e:
        logging.error(f"GPT error: {e}")
        return f"❌ GPT error: {e}"

# ─── SIMILARITY SCORING ────────────────────────────────────────────
def compute_similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        ea = model.encode(a, convert_to_tensor=True)
        eb = model.encode(b, convert_to_tensor=True)
        sim = util.cos_sim(ea, eb).item()
        return round(sim * 100, 2)
    except Exception:
        return 0.0

# ─── FACT-CHECKING POSTS ─────────────────────────────────────────
def fact_check_post(post: str, ref: str) -> str:
    prompt = (
        "You're a fact-checker. For each medical claim in the POST:\n"
        "✅ if supported, ❌ if contradicted, ⚠️ if unverifiable\n\n"
        f"POST:\n\"\"\"{post}\"\"\"\n\nREFERENCE:\n\"\"\"{ref[:8000]}\"\"\"\n\nBegin analysis."
    )
    msgs = [
        {"role":"system","content":"Carefully compare claims to reference."},
        {"role":"user","content":prompt}
    ]
    try:
        r = client.chat.completions.create(
            model="gpt-4o", messages=msgs, temperature=0.2, max_tokens=800
        )
        return r.choices[0].message.content
    except Exception as e:
        logging.error(f"Fact-check error: {e}")
        return f"❌ Fact-check error: {e}"

# ─── SAVE FEEDBACK ────────────────────────────────────────────────
def save_feedback(record: dict):
    row = [
        record.get("Timestamp",""),
        record.get("Input",""),
        record.get("Type",""),
        record.get("Source",""),
        record.get("Summary",""),
        record.get("Feedback",""),
        record.get("Similarity Score","")
    ]
    try:
        feedback_sheet.append_row(row)
        st.success("✅ Feedback saved.")
    except Exception as e:
        logging.error(f"Save feedback error: {e}")
        st.error(f"Could not save feedback: {e}")

# ─── STREAMLIT UI ────────────────────────────────────────────────

st.sidebar.image(
    "https://www.bcu.ac.uk/images/default-source/marketing/logos/bcu-logo.svg", width=150
)
st.sidebar.title("🧠 Medicine Info Validator")
st.sidebar.warning(
    "**Disclaimer:** For general understanding only; not a substitute for medical advice."
)
st.sidebar.caption("Developed by Doaa Al-Turkey")
st.sidebar.markdown("---")

st.header("Validate Medicine Information")

input_type = st.radio(
    "Analyze as:", ["Medicine","Medical Query","Webpage with Medical Claims"],
    horizontal=True
)

placeholder = {
    "Medicine":"e.g. Paracetamol",
    "Medical Query":"e.g. symptoms of flu",
    "Webpage with Medical Claims":"e.g. https://example.com/article"
}[input_type]

user_input = st.text_input(f"Enter {input_type}:", placeholder)

if 'done' not in st.session_state:
    st.session_state.done = False
    st.session_state.ref = ""
    st.session_state.summary = ""
    st.session_state.source = ""

if st.button("Analyze"):
    st.session_state.done = False
    st.session_state.ref = ""
    st.session_state.summary = ""
    st.session_state.source = ""
    if not user_input.strip():
        st.warning("Please provide valid input.")
    else:
        with st.status(f"Fetching {input_type} data...", expanded=True) as status_ui:
            ref, src = "", ""
            if input_type=="Medicine":
                bnf = fetch_bnf_info(user_input)
                ref = bnf["full_text"] or bnf["search_summary"]
                src = "BNF"
            elif input_type=="Medical Query":
                ref = fetch_nhs_info(user_input)
                src = "NHS"
            else:
                if not user_input.startswith("http"):
                    st.error("URL must start with http:// or https://")
                else:
                    ref = fetch_url_content(user_input)
                    src = "Webpage"

            st.session_state.ref = ref
            st.session_state.source = src

            if not ref.strip():
                status_ui.update(label=f"No data from {src}.", state="error")
            else:
                status_ui.update(label="Generating summary...", state="running")
                summ = generate_response(user_input, ref)
                st.session_state.summary = summ
                st.session_state.done = True
                status_ui.update(label="Complete!", state="complete")

if st.session_state.done:
    st.markdown(f"### 🤖 Summary (source: {st.session_state.source})")
    st.write(st.session_state.summary)

    if st.session_state.source in ["NHS","Webpage"]:
        sim = compute_similarity_score(st.session_state.summary, st.session_state.ref)
        st.markdown(f"**Similarity Score:** {sim}/100")

    if input_type=="Webpage":
        st.markdown("### 🧾 Webpage Claim Verification")
        with st.spinner("Fact-checking..."):
            chk = fact_check_post(st.session_state.ref, st.session_state.ref)
        st.write(chk)

    st.markdown("---")
    with st.expander("📝 Doctor Feedback (optional)"):
        fb = st.text_area("Your feedback:")
        if st.button("Submit Feedback"):
            if fb.strip():
                rec = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input": user_input,
                    "Type": input_type,
                    "Source": st.session_state.source,
                    "Summary": st.session_state.summary,
                    "Feedback": fb
                }
                if st.session_state.source in ["NHS","Webpage"]:
                    rec["Similarity Score"] = sim
                save_feedback(rec)
            else:
                st.warning("Enter feedback before submitting.")

    if st.session_state.source in ["BNF","NHS"]:
        st.markdown("## 🔍 Fact-Check Social Media")
        tabs = st.tabs(["🐦 Twitter","💬 Reddit"])

        with tabs[0]:
            st.markdown(f"**Tweets for '{user_input}'**")
            try:
                tweets = twitter_client.search_recent_tweets(
                    query=f"{user_input} -is:retweet lang:en",
                    max_results=5, tweet_fields=["text"]
                )
                for i,t in enumerate(tweets.data or []):
                    txt = t.text
                    st.markdown(f"**Tweet {i+1}:**")
                    st.info(txt)
                    with st.spinner("Fact-checking..."):
                        res = fact_check_post(txt, st.session_state.ref)
                        sc = compute_similarity_score(txt, st.session_state.ref)
                    st.write(res)
                    st.markdown(f"**Similarity:** {sc}/100")
            except Exception as e:
                st.warning(f"Twitter error: {e}")

        with tabs[1]:
            st.markdown(f"**Reddit for '{user_input}'**")
            try:
                posts = reddit.subreddit("medicine").search(user_input, limit=5, sort="new")
                for i,post in enumerate(posts):
                    txt = post.title+"\n\n"+post.selftext
                    st.markdown(f"**Post {i+1}:** {post.title}")
                    with st.spinner("Fact-checking..."):
                        res = fact_check_post(txt, st.session_state.ref)
                        sc = compute_similarity_score(txt, st.session_state.ref)
                    st.write(res)
                    st.markdown(f"**Similarity:** {sc}/100")
            except Exception as e:
                st.warning(f"Reddit error: {e}")

footer()
