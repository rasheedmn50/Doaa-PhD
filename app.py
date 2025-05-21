# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import time
from urllib.parse import quote
import tweepy
import praw
import numpy as np
import logging
import json
import gspread
from google.oauth2.service_account import Credentials

# ─── LOGGING ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────
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
    OPENAI_API_KEY            = st.secrets["OPENAI_API_KEY"]
    REDDIT_CLIENT_ID          = st.secrets["REDDIT_CLIENT_ID"]
    REDDIT_CLIENT_SECRET      = st.secrets["REDDIT_CLIENT_SECRET"]
    REDDIT_USER_AGENT         = st.secrets["REDDIT_USER_AGENT"]
    TWITTER_BEARER_TOKEN      = st.secrets["TWITTER_BEARER_TOKEN"]
    GOOGLE_SHEETS_CREDENTIALS = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    GOOGLE_SHEET_NAME         = st.secrets["GOOGLE_SHEET_NAME"]
    UK_PROXY                  = st.secrets.get("UK_PROXY", None)

    gcred = json.loads(GOOGLE_SHEETS_CREDENTIALS)
    scopes = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(gcred, scopes=scopes)
    gc = gspread.authorize(creds)
    feedback_sheet = gc.open(GOOGLE_SHEET_NAME).sheet1

except KeyError as e:
    st.error(f"🔴 Missing secret: {e}")
    st.stop()
except Exception as e:
    st.error(f"🔴 Google Sheets init error: {e}")
    st.stop()

# ─── INIT CLIENTS ─────────────────────────────────────────────────
try:
    client         = OpenAI(api_key=OPENAI_API_KEY)
    model          = SentenceTransformer("all-MiniLM-L6-v2")
    reddit         = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"🔴 API client init error: {e}")
    st.stop()

# ─── UK-ROUTED REQUESTS SESSION ────────────────────────────────────
def make_uk_session():
    s = requests.Session()
    if UK_PROXY:
        s.proxies.update({"http": UK_PROXY, "https": UK_PROXY})
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9"
    })
    return s

# ─── BNF SCRAPER ──────────────────────────────────────────────────
def fetch_bnf_info(med_name: str, max_links: int = 5):
    base = "https://bnf.nice.org.uk"
    search_url = f"{base}/search/?q={quote(med_name)}"
    sess = make_uk_session()
    out = {"card_snippets": [], "links": [], "full_text": ""}

    try:
        r = sess.get(search_url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("div[data-component='card']")[:max_links]
        for card in cards:
            a = card.find("a", href=True)
            if not a: continue
            href = a["href"]
            url  = href if href.startswith("http") else base + href
            title = a.get_text(strip=True)
            snippet = card.get_text(" ", strip=True)
            out["links"].append({"title": title, "url": url})
            out["card_snippets"].append(snippet)

        for link in out["links"]:
            time.sleep(0.5)
            p = sess.get(link["url"], timeout=10); p.raise_for_status()
            ps = BeautifulSoup(p.text, "html.parser")
            topic = ps.find(id="topic") or ps.find("main") or ps.body
            text = topic.get_text("\n", strip=True) if topic else ps.get_text("\n", strip=True)
            out["full_text"] += f"## {link['title']}\n\n{text}\n\n"

    except Exception as e:
        logging.error(f"BNF fetch error: {e}")

    return out

# ─── NHS SCRAPER ──────────────────────────────────────────────────
def fetch_nhs_info(query: str, min_len: int = 1500, max_results: int = 5) -> str:
    base = "https://www.nhs.uk"
    search_url = f"{base}/search/results"
    sess = make_uk_session()
    compiled = ""

    try:
        resp = sess.get(search_url, params={"q": query}, timeout=10); resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results_ul = soup.find("ul", class_="nhsuk-list")
        links = [a["href"] for a in results_ul.find_all("a", href=True)[:max_results]] if results_ul else []
        for href in links:
            url = href if href.startswith("http") else base + href
            pr = sess.get(url, timeout=10); pr.raise_for_status()
            ps = BeautifulSoup(pr.text, "html.parser")
            title = ps.find("h2"); paras = ps.find_all("p", attrs={"data-block-key": True})
            txt = "\n".join(p.get_text(strip=True) for p in paras)
            compiled += f"{title.get_text(strip=True) if title else ''}\n{txt}\n\n"
            if len(compiled) >= min_len: break
    except Exception as e:
        logging.error(f"NHS fetch error: {e}")

    return compiled

# ─── URL FETCH ────────────────────────────────────────────────────
def fetch_url_content(url: str) -> str:
    sess = make_uk_session()
    try:
        r = sess.get(url, timeout=15); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        main = soup.find("article") or soup.find("main") or soup.body
        return main.get_text("\n", strip=True) if main else ""
    except Exception as e:
        logging.error(f"URL fetch failed: {e}")
        return ""

# ─── GPT-4 SUMMARY ────────────────────────────────────────────────
def generate_response(inp: str, ref: str) -> str:
    prompt = (
        "You are a medical assistant. Use *only* the reference below.\n\n"
        f"User query: {inp}\n\nReference:\n{ref[:12000]}"
    )
    msgs = [
        {"role":"system","content":"Answer strictly from the reference."},
        {"role":"user",  "content":prompt}
    ]
    try:
        r = client.chat.completions.create(
            model="gpt-4o", messages=msgs, temperature=0.3, max_tokens=1000
        )
        return r.choices[0].message.content
    except Exception as e:
        logging.error(f"GPT error: {e}")
        return f"❌ GPT error: {e}"

# ─── SIMILARITY SCORE ─────────────────────────────────────────────
def compute_similarity_score(a: str, b: str) -> float:
    if not a or not b: return 0.0
    try:
        ea = model.encode(a, convert_to_tensor=True)
        eb = model.encode(b, convert_to_tensor=True)
        sim = util.cos_sim(ea, eb).item()
        return round(sim * 100, 2)
    except Exception:
        return 0.0

# ─── FACT-CHECK POSTS ──────────────────────────────────────────────
def fact_check_post(post: str, ref: str) -> str:
    prompt = (
        "Fact-check this post against the reference below.\n"
        "✅ supported, ❌ contradicted, ⚠️ unverifiable.\n\n"
        f"POST:\n\"\"\"{post}\"\"\"\n\nREFERENCE:\n\"\"\"{ref[:8000]}\"\"\"\n\nBegin."
    )
    msgs = [
        {"role":"system","content":"Fact-check medical claims."},
        {"role":"user",  "content":prompt}
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
def save_feedback(rec: dict):
    row = [
        rec.get("Timestamp",""),
        rec.get("Input",""),
        rec.get("Type",""),
        rec.get("Source",""),
        rec.get("Summary",""),
        rec.get("Feedback",""),
        rec.get("Similarity Score","")
    ]
    try:
        feedback_sheet.append_row(row)
        st.success("✅ Feedback saved.")
    except Exception as e:
        logging.error(f"Save feedback failed: {e}")
        st.error(f"Could not save feedback: {e}")

# ─── STREAMLIT UI ────────────────────────────────────────────────
st.sidebar.image(
    "https://www.bcu.ac.uk/images/default-source/marketing/logos/bcu-logo.svg", width=150
)
st.sidebar.title("🧠 Medicine Info Validator")
st.sidebar.warning("Disclaimer: For general understanding only; not medical advice.")
st.sidebar.caption("Developed by Doaa Al-Turkey")
st.sidebar.markdown("---")

st.header("Validate Medicine Information")

input_type = st.radio(
    "Analyze as:", ["Medicine","Medical Query","Webpage with Medical Claims"],
    horizontal=True
)

placeholders = {
    "Medicine": "e.g. Paracetamol",
    "Medical Query": "e.g. symptoms of flu",
    "Webpage with Medical Claims": "e.g. https://example.com/article"
}
user_input = st.text_input(f"Enter {input_type}:", placeholders[input_type])

if 'done' not in st.session_state:
    st.session_state.update({"done":False,"ref":"","summary":"","source":""})

if st.button("Analyze"):
    st.session_state.update({"done":False,"ref":"","summary":"","source":""})
    if not user_input.strip():
        st.warning("Please enter valid input.")
    else:
        with st.status(f"Fetching {input_type}…", expanded=True) as status_ui:
            ref, src = "", ""
            if input_type == "Medicine":
                bnf = fetch_bnf_info(user_input)
                if bnf["full_text"].strip():
                    ref = bnf["full_text"]
                elif bnf["card_snippets"]:
                    ref = "\n\n".join(bnf["card_snippets"])
                else:
                    st.error(f"❌ No BNF info for “{user_input}”."); status_ui.error("No BNF content."); st.session_state.done=True; st.stop() 
                src = "BNF"

            elif input_type == "Medical Query":
                ref = fetch_nhs_info(user_input)
                src = "NHS"

            else:  # Webpage with Medical Claims
                # 1) raw page text
                page_text = fetch_url_content(user_input)
                # 2) summary
                status_ui.update("Summarizing webpage…")
                summ_r = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role":"system","content":"You are a medical summarisation assistant."},
                        {"role":"user","content":(
                            "Summarize the following page text in layman terms:\n\n" +
                            page_text[:15000]
                        )}
                    ],
                    temperature=0.2, max_tokens=500
                )
                webpage_summary = summ_r.choices[0].message.content
                ref = webpage_summary
                src = "Webpage"

            st.session_state.update(ref=ref, source=src)

            status_ui.update("Generating summary…")
            summ = generate_response(user_input, ref)
            st.session_state.update(summary=summ, done=True)
            status_ui.success("Complete!")

if st.session_state.done:
    st.markdown(f"### 🤖 Summary ({st.session_state.source})")
    st.write(st.session_state.summary)

    if input_type in ["Medicine","Medical Query"]:
        sim = compute_similarity_score(
            st.session_state.summary, st.session_state.ref
        )
        st.markdown(f"**Similarity Score:** {sim}/100")

    if input_type == "Webpage with Medical Claims":
        st.markdown("### 📄 Webpage Summary")
        st.write(st.session_state.ref)

        st.markdown("### ✅⚠️❌ Claims Verification", unsafe_allow_html=True)
        with st.spinner("Verifying claims…"):
            verify_r = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"You are a medical fact-checking assistant."},
                    {"role":"user","content":(
                        "Below is a webpage summary and its raw text. "
                        "For each medical claim, output an HTML paragraph "
                        "with a green span for ✅ verified, orange span for ⚠️ unverified, "
                        "red span for ❌ contradicted.\n\n"
                        f"SUMMARY:\n{st.session_state.ref}\n\n"
                        f"RAW:\n{fetch_url_content(user_input)[:15000]}"
                    )}
                ],
                temperature=0.0, max_tokens=800
            )
        st.markdown(verify_r.choices[0].message.content, unsafe_allow_html=True)

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
                if input_type in ["Medicine","Medical Query"]:
                    rec["Similarity Score"] = sim
                save_feedback(rec)
            else:
                st.warning("Enter feedback before submitting.")

footer()
