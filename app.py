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
    except Exception as e:
        logging.error(f"BNF search failed: {e}")
        return out

    soup = BeautifulSoup(r.text, "html.parser")

    # pull card elements
    cards = soup.select("div[data-component='card']")[:max_links]
    for card in cards:
        a = card.find("a", href=True)
        if not a: continue
        href = a["href"]
        title = a.get_text(strip=True)
        url = href if href.startswith("http") else base + href
        out["links"].append({"title": title, "url": url})
        snippet = card.get_text(" ", strip=True)
        out["card_snippets"].append(snippet)

    # if we found detail pages, fetch them
    for link in out["links"]:
        try:
            time.sleep(0.5)
            p = sess.get(link["url"], timeout=10); p.raise_for_status()
            ps = BeautifulSoup(p.text, "html.parser")
            topic = ps.find(id="topic") or ps.find("main") or ps.body
            text = topic.get_text("\n", strip=True) if topic else ps.get_text("\n", strip=True)
            out["full_text"] += f"## {link['title']}\n\n{text}\n\n"
        except Exception as e:
            logging.warning(f"BNF detail fetch failed {link['url']}: {e}")

    return out

# ─── NHS SCRAPER ──────────────────────────────────────────────────
def fetch_nhs_info(query: str, min_len: int = 1500, max_results: int = 5) -> str:
    base = "https://www.nhs.uk"
    search_url = f"{base}/search/results"
    sess = make_uk_session()
    compiled = ""

    try:
        resp = sess.get(search_url, params={"q": query, "page": 0}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"NHS search failed: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    results_ul = soup.find("ul", class_="nhsuk-list")
    links = []
    if results_ul:
        for a in results_ul.find_all("a", href=True)[:max_results]:
            href = a["href"]
            full = href if href.startswith("http") else base + href
            links.append(full)

    for url in links:
        try:
            time.sleep(0.5)
            pr = sess.get(url, timeout=10); pr.raise_for_status()
            ps = BeautifulSoup(pr.text, "html.parser")
            h2 = ps.find("h2"); title = h2.get_text(strip=True) if h2 else ""
            paras = ps.find_all("p", attrs={"data-block-key": True})
            txt = "\n".join(p.get_text(strip=True) for p in paras)
            compiled += f"{title}\n{txt}\n\n"
            if len(compiled) >= min_len:
                break
        except Exception as e:
            logging.warning(f"NHS detail fetch failed {url}: {e}")

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

# ─── GPT-4o SUMMARY ───────────────────────────────────────────────
def generate_response(inp: str, ref: str) -> str:
    prompt = (
        "You are a medical assistant. Use *only* the reference below.\n\n"
        f"User query: {inp}\n\nReference:\n{ref[:12000]}"
    )
    msgs = [
        {"role": "system", "content": "Answer strictly from the reference."},
        {"role": "user",   "content": prompt}
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
    if not a or not b:
        return 0.0
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
        {"role": "system", "content": "Fact-check medical claims."},
        {"role": "user",   "content": prompt}
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
st.sidebar.warning(
    "Disclaimer: For general understanding only; not medical advice."
)
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
    st.session_state.update({
        "done": False,
        "ref": "",
        "summary": "",
        "source": ""
    })

if st.button("Analyze"):
    st.session_state.update({"done":False,"ref":"","summary":"","source":""})
    if not user_input.strip():
        st.warning("Please enter valid input.")
    else:
        with st.status(f"Fetching {input_type}…", expanded=True) as status_ui:
            ref_text, src = "", ""

            if input_type == "Medicine":
                bnf = fetch_bnf_info(user_input)
                # only accept detail or snippet—no generic subscription text
                if bnf["full_text"].strip():
                    ref_text = bnf["full_text"]
                elif bnf["card_snippets"]:
                    ref_text = "\n\n".join(bnf["card_snippets"])
                else:
                    st.error(f"❌ No medicine information found for “{user_input}” on BNF.")
                    status_ui.update(label="No BNF content.", state="error")
                    continue
                src = "BNF"

            elif input_type == "Medical Query":
                ref_text = fetch_nhs_info(user_input)
                src = "NHS"

            else:  # Webpage with Medical Claims
                page_text = fetch_url_content(user_input)
                # derive NHS query from page title if possible
                title = ""
                try:
                    html = requests.get(user_input, timeout=10).text
                    soup = BeautifulSoup(html,"html.parser")
                    title = soup.title.get_text(strip=True) if soup.title else ""
                except:
                    pass
                ref_text = fetch_nhs_info(title or user_input)
                src = "NHS"

            st.session_state["ref"]    = ref_text
            st.session_state["source"] = src

            status_ui.update(label="Generating summary…", state="running")
            summ = generate_response(user_input, ref_text)
            st.session_state["summary"] = summ
            st.session_state["done"]    = True
            status_ui.update(label="Complete!", state="complete")

if st.session_state["done"]:
    st.markdown(f"### 🤖 Summary (based on {st.session_state['source']})")
    st.write(st.session_state["summary"])

    if input_type in ["Medicine","Medical Query"]:
        sim = compute_similarity_score(
            st.session_state["summary"],st.session_state["ref"]
        )
        st.markdown(f"**Similarity Score:** {sim}/100")

    if input_type == "Webpage with Medical Claims":
        st.markdown("### 🔎 Webpage Medical-Claim Verification")
        with st.spinner("Fact-checking claims…"):
            result = fact_check_post(
                fetch_url_content(user_input),
                st.session_state["ref"]
            )
        st.write(result)

    st.markdown("---")
    with st.expander("📝 Doctor Feedback (optional)"):
        fb = st.text_area("Your feedback:")
        if st.button("Submit Feedback"):
            if fb.strip():
                rec = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input": user_input,
                    "Type": input_type,
                    "Source": st.session_state["source"],
                    "Summary": st.session_state["summary"],
                    "Feedback": fb
                }
                if input_type in ["Medicine","Medical Query"]:
                    rec["Similarity Score"] = sim
                save_feedback(rec)
            else:
                st.warning("Enter feedback before submitting.")

    if input_type in ["Medicine","Medical Query"]:
        st.markdown("## 🔍 Fact-Check Social Media")
        tabs = st.tabs(["🐦 Twitter","💬 Reddit"])

        with tabs[0]:
            st.markdown(f"**Tweets for '{user_input}'**")
            try:
                tw = twitter_client.search_recent_tweets(
                    query=f"{user_input} -is:retweet lang:en",
                    max_results=5, tweet_fields=["text"]
                )
                for i,t in enumerate(tw.data or []):
                    txt = t.text
                    st.markdown(f"**Tweet {i+1}:**"); st.info(txt)
                    with st.spinner("Fact-checking…"):
                        fc = fact_check_post(txt, st.session_state["ref"])
                        sc = compute_similarity_score(txt, st.session_state["ref"])
                    st.write(fc); st.markdown(f"**Similarity:** {sc}/100")
            except Exception as e:
                st.warning(f"Twitter error: {e}")

        with tabs[1]:
            st.markdown(f"**Reddit for '{user_input}'**")
            try:
                posts = reddit.subreddit("medicine").search(user_input, limit=5, sort="new")
                for i,p in enumerate(posts):
                    txt = p.title + "\n\n" + p.selftext
                    st.markdown(f"**Post {i+1}:** {p.title}")
                    with st.spinner("Fact-checking…"):
                        fc = fact_check_post(txt, st.session_state["ref"])
                        sc = compute_similarity_score(txt, st.session_state["ref"])
                    st.write(fc); st.markdown(f"**Similarity:** {sc}/100")
            except Exception as e:
                st.warning(f"Reddit error: {e}")

footer()
