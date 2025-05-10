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
import json # For parsing Google Service Account JSON

# For Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== API KEYS & SECRETS ==========
# These will be set in Streamlit Cloud's secrets management
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
    st.error(f"🔴 Missing secret: '{e}'. Please ensure all API keys and Google Sheets credentials are correctly set in your Streamlit Cloud app secrets.")
    st.stop()
except Exception as e:
    st.error(f"🔴 Error loading secrets or initializing Google Sheets: {e}")
    logging.error(f"Error loading secrets or initializing Google Sheets: {e}")
    st.stop()

# ========== INIT API CLIENTS ==========
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"Error initializing API clients (OpenAI, SentenceTransformer, PRAW, Tweepy): {e}. Please check API keys and configurations if applicable.")
    logging.error(f"Error initializing API clients: {e}")
    st.stop()

# ========== HELPER FUNCTIONS ==========
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import logging
import time

def fetch_bnf_info(medicine_name: str, max_links: int = 5):
    """
    Fetch BNF search results and detail pages without Selenium.
    Returns a dict with:
      - search_summary: first few lines from the search page
      - links: list of {title, url}
      - full_text: concatenated text from each detail page
    """
    base_url = "https://bnf.nice.org.uk"
    search_url = f"{base_url}/search/?q={quote(medicine_name)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
    }

    results = {
        "search_summary": "",
        "links": [],
        "full_text": ""
    }

    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"BNF search request failed: {e}")
        return results

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) grab a quick summary from the search page (first 20 lines of the <main> block)
    main = soup.find("main", id="maincontent") or soup.find("main")
    if main:
        lines = [ln.strip() for ln in main.get_text("\n").splitlines() if ln.strip()]
        results["search_summary"] = "\n".join(lines[:20])

    # 2) find up to max_links cards
    cards = soup.select("header.card__header a[href]")[:max_links]
    for a in cards:
        href = a["href"]
        title = a.get_text(strip=True)
        url = href if href.startswith("http") else base_url + href
        results["links"].append({"title": title, "url": url})

    # 3) fetch each detail page in turn
    for link in results["links"]:
        try:
            time.sleep(0.5)  # be polite
            page = requests.get(link["url"], headers=headers, timeout=10)
            page.raise_for_status()
            psoup = BeautifulSoup(page.text, "html.parser")
            # Prefer the <div id="topic"> or else dump all text under <main>
            topic = psoup.find(id="topic") or psoup.find("main") or psoup.body
            text = topic.get_text("\n", strip=True) if topic else psoup.get_text("\n", strip=True)
            results["full_text"] += f"## {link['title']}\n\n{text}\n\n"
        except Exception as e:
            logging.warning(f"Failed to fetch BNF detail page {link['url']}: {e}")

    return results

def fetch_nhs_info(query):
    logging.info(f"Fetching NHS info for query: {query}")
    base_url = "https://www.nhs.uk"
    search_url = f"{base_url}/search/"
    params = {"q": query}
    content = ""
    try:
        with requests.Session() as session:
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            response = session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results_links = soup.select('div.nhsuk-list-panel__item a.nhsuk-list-panel__link', limit=5)
            if not results_links:
                results_links = soup.select('a[href*="/conditions/"], a[href*="/medicines/"]', limit=5)

            logging.info(f"Found {len(results_links)} links from NHS search for query '{query}'.")
            if not results_links:
                st.warning(f"No direct links found on NHS for '{query}'. Content from the search page itself might be limited.")
                main_content_area = soup.find("main", id="maincontent")
                if main_content_area:
                    return main_content_area.get_text(separator="\n", strip=True) + "\n"
                return soup.get_text(separator="\n", strip=True) + "\n"

            fetched_urls = set()
            for link_tag in results_links:
                href = link_tag.get("href")
                if not href: continue

                if href.startswith("http"): page_url = href
                elif href.startswith("/"): page_url = base_url + href
                else:
                    logging.warning(f"Skipping malformed or external NHS link: {href}")
                    continue
                
                if page_url in fetched_urls: continue

                logging.info(f"Fetching NHS page: {page_url}")
                try:
                    page_resp = session.get(page_url, timeout=10)
                    page_resp.raise_for_status()
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    
                    article_content = page_soup.find("article") or page_soup.find("main")
                    if article_content:
                        content += article_content.get_text(separator="\n", strip=True) + "\n\n"
                    else:
                        content += page_soup.get_text(separator="\n", strip=True) + "\n\n"
                    fetched_urls.add(page_url)
                    time.sleep(0.5) 
                except requests.exceptions.RequestException as e_page:
                    logging.error(f"Error fetching NHS detail page {page_url}: {e_page}")
            
    except requests.exceptions.RequestException as e_search:
        logging.error(f"Error fetching NHS search page for query '{query}': {e_search}")
        st.error(f"Could not connect to NHS website or search failed: {e_search}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during NHS info fetching: {e}")
        st.error(f"An unexpected error occurred while fetching NHS data: {e}")

    if not content: logging.warning("NHS scraping returned no content for query '%s'.", query)
    return content

def fetch_url_content(url):
    logging.info(f"Fetching content for URL: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_content = soup.find('article') or soup.find('main')
        if main_content: text = main_content.get_text(separator="\n", strip=True)
        else: text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text

    except requests.exceptions.MissingSchema:
        logging.error(f"Invalid URL (Missing Schema): {url}. Please include http:// or https://")
        st.error(f"Invalid URL: {url}. Please ensure it starts with http:// or https://")
        return ""
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error for URL: {url}. Check the URL and your internet connection.")
        st.error(f"Connection error: Could not connect to {url}.")
        return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        st.error(f"Failed to fetch content from URL {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching URL content for {url}: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return ""

def generate_response(user_input_topic, reference_text):
    max_ref_length = 12000 
    truncated_ref_text = reference_text[:max_ref_length]
    
    if not truncated_ref_text.strip():
        return "Reference information was empty or too short to process for AI summary."

    prompt = (
        f"You are a medical information assistant. Your goal is to provide clear, concise, and layman-friendly information based *only* on the verified medical reference text provided below.\n"
        f"Analyze the following reference information in relation to the topic: '{user_input_topic}'.\n\n"
        f"REFERENCE INFORMATION (first {max_ref_length} characters):\n{truncated_ref_text}\n\n"
        f"TASK: Based strictly on the reference information, provide a comprehensive summary or answer related to '{user_input_topic}'. "
        f"If the reference text does not contain relevant information for '{user_input_topic}', explicitly state that. "
        f"Do not invent information or use external knowledge. Focus on accuracy and clarity for a general audience. "
        f"Structure your response clearly. If discussing a medicine, cover key aspects like uses, important warnings, and common side effects if mentioned in the reference."
    )
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant that bases answers strictly on provided reference text, making it understandable to a layperson."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages, temperature=0.2, max_tokens=1000, top_p=0.95,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return f"Error generating response from AI: {e}"

def compute_similarity_score(text_a, text_b):
    if not text_a or not text_b: return 0.0
    try:
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        similarity = util.cos_sim(emb_a, emb_b).item()
        score = round(max(0, similarity * 100), 2) 
        return score
    except Exception as e:
        logging.error(f"Error computing similarity score: {e}")
        return 0.0

def fact_check_post(post_text, reference_text):
    max_ref_length = 8000 
    truncated_ref_text = reference_text[:max_ref_length]

    if not post_text.strip(): return "Cannot fact-check an empty post."
    if not truncated_ref_text.strip(): return "Cannot fact-check against empty reference information."

    prompt = (
        "You are a meticulous medical fact-checking assistant. You will be given a POST (e.g., from social media or a webpage segment) "
        "and verified REFERENCE medical information.\n"
        "Your task is to:\n"
        "1. Carefully read the POST and identify distinct factual claims related to health or medicine.\n"
        "2. For each claim, compare it against the provided REFERENCE information.\n"
        "3. Mark each claim with ONE of the following emojis:\n"
        "   - ✅ (Supported): If the claim is directly supported or aligns with the REFERENCE information.\n"
        "   - ❌ (Contradicted): If the claim contradicts or is inconsistent with the REFERENCE information.\n"
        "   - ⚠️ (Not Verifiable/Not Addressed): If the claim is not addressed, cannot be verified, or is too vague based *solely* on the provided REFERENCE information. Do not use external knowledge.\n"
        "4. Provide a brief explanation for your marking, quoting or referring to parts of the REFERENCE if helpful.\n"
        "5. If the POST contains no specific medical claims, state that.\n\n"
        f"POST TO ANALYZE:\n\"\"\"{post_text}\"\"\"\n\n"
        f"VERIFIED REFERENCE INFORMATION (first {max_ref_length} characters):\n\"\"\"{truncated_ref_text}\"\"\"\n\n"
        "Begin your analysis:"
    )
    messages = [{"role": "system", "content": "You are a fact-checking AI specializing in medical claims, comparing a post against provided reference text."},
                {"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1, 
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call for fact-checking failed: {e}")
        return f"Error during fact-checking process: {e}"

def save_feedback(record):
    try:
        row_to_insert = [
            record.get("Timestamp", ""), record.get("Input", ""), record.get("Type", ""),
            record.get("Source", ""), record.get("Summary", ""), record.get("Feedback", ""),
            record.get("Similarity Score (Summary vs Source)", "") 
        ]
        feedback_sheet.append_row(row_to_insert)
        logging.info("Feedback saved successfully to Google Sheets.")
        st.success("✅ Feedback recorded. Thank you!")
    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API Error: {e}")
        st.error(f"Could not save feedback to Google Sheets due to API error: {e}. Check share permissions for the service account and API enablement in GCP.")
    except Exception as e:
        logging.error(f"Failed to save feedback to Google Sheets: {e}")
        st.error(f"Could not save feedback: {e}")

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="Medicine Info Validator", layout="wide", page_icon="🧠")

with st.sidebar:
    st.image("https://www.bcu.ac.uk/images/default-source/marketing/logos/bcu-logo.svg", width=150)
    st.title("🧠 Medicine Info Validator")
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This tool provides information for general understanding and "
        "is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider "
        "with any questions you may have regarding a medical condition."
    )
    st.markdown("---")
    st.caption(f"Developed by **Doaa Al-Turkey**")

default_session_state = {
    'analysis_done': False, 'reference_text': "", 'generated_summary': "",
    'source_name': "", 'user_input_cache': "", 'bnf_results_cache': None
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.header("Analyze Medical Information")

input_type = st.radio(
    "What are you analyzing?",
    ["Medicine", "Medical Query", "Webpage with Medical Claims"],
    horizontal=True, key="input_type_radio",
    captions = ["e.g., Paracetamol (from BNF search)", "e.g., symptoms of flu (from NHS)", "e.g., URL of an article"]
)

user_input_label = "Enter Medicine Name:"
example_placeholder = "e.g., Amoxicillin, Ibuprofen"
if input_type == "Medical Query":
    user_input_label = "Enter Medical Query:"
    example_placeholder = "e.g., treatment for common cold, side effects of aspirin"
elif input_type == "Webpage with Medical Claims":
    user_input_label = "Enter Webpage URL:"
    example_placeholder = "e.g., https://www.example.com/medical-article"

user_input = st.text_input(
    user_input_label, placeholder=example_placeholder,
    key="user_main_input", value=st.session_state.user_input_cache
)

col1, col2 = st.columns([1,5])
with col1:
    analyze_button_pressed = st.button("Analyze", key="analyze_button", type="primary", use_container_width=True)
with col2:
    if st.button("Clear Results", key="clear_button", use_container_width=True):
        for key in ['analysis_done', 'reference_text', 'generated_summary', 'source_name', 'bnf_results_cache']:
            st.session_state[key] = default_session_state[key]
        # st.session_state.user_input_cache = "" # Uncomment to also clear the input field
        st.rerun()

if analyze_button_pressed:
    st.session_state.analysis_done = False
    st.session_state.reference_text = ""
    st.session_state.generated_summary = ""
    st.session_state.source_name = ""
    st.session_state.user_input_cache = user_input
    st.session_state.bnf_results_cache = None # Reset BNF cache

    if not user_input.strip():
        st.warning("Please enter a valid input.")
    else:
        with st.status(f"Analyzing '{user_input[:30]}...' ({input_type})", expanded=True) as status_ui:
            reference_text_for_ai = "" # Text to be used by AI and for social media checks
            source_name = ""
            
            status_ui.update(label=f"Fetching information for {input_type}...")
            if input_type == "Medicine":
                bnf_results = fetch_bnf_info(user_input)
                st.session_state.bnf_results_cache = bnf_results # Store the full dict
                reference_text_for_ai = bnf_results.get("search_page_summary", "")
                source_name = "BNF Search Results"
                # Error/warning messages for BNF are now handled within fetch_bnf_info itself
                # We proceed if we have at least links or a summary
                if not reference_text_for_ai and not bnf_results.get("links"):
                    status_ui.update(label=f"No information found from {source_name}.", state="error", expanded=False)
                else: # We have something from BNF
                    status_ui.update(label=f"Retrieved search info from {source_name}.")

            elif input_type == "Medical Query":
                reference_text_for_ai = fetch_nhs_info(user_input)
                source_name = "NHS"
            else: # Webpage
                if not (user_input.startswith("http://") or user_input.startswith("https://")):
                    st.error("Please enter a valid URL starting with http:// or https://")
                    status_ui.update(label="Invalid URL provided.", state="error", expanded=False)
                    reference_text_for_ai = None # Signal error
                else:
                    reference_text_for_ai = fetch_url_content(user_input)
                    source_name = "Webpage"
            
            st.session_state.source_name = source_name
            st.session_state.reference_text = reference_text_for_ai # This is key for social media checks

            if reference_text_for_ai is None: # Error case like invalid URL
                pass # Error already shown by status_ui or fetch function
            elif not reference_text_for_ai.strip():
                if input_type != "Medicine": # For Medicine, an empty summary but existing links is okay
                    st.error(f"❌ Could not retrieve reference information from {source_name} for '{user_input}'.")
                    status_ui.update(label=f"Failed to get data from {source_name}.", state="error", expanded=False)
                # For Medicine, if reference_text_for_ai is empty but links exist, it's handled in display
                st.session_state.generated_summary = "" # No AI summary if no reference text
                st.session_state.analysis_done = True # Still mark as done to show results (e.g., BNF links)
                if not (input_type == "Medicine" and st.session_state.bnf_results_cache and st.session_state.bnf_results_cache.get("links")):
                    status_ui.update(label=f"No textual content for AI summary from {source_name}.", state="warning", expanded=False)
                else:
                     status_ui.update(label=f"Analysis of {source_name} complete.", state="complete", expanded=False)

            else: # We have reference_text_for_ai
                status_ui.update(label=f"Generating AI summary based on {source_name} data...")
                st.session_state.generated_summary = generate_response(user_input, reference_text_for_ai)
                st.session_state.analysis_done = True
                status_ui.update(label="Analysis complete!", state="complete", expanded=False)

if st.session_state.analysis_done:
    st.markdown("---")
    st.subheader(f"Analysis Results for: \"{st.session_state.user_input_cache}\"")

    if input_type == "Medicine" and st.session_state.get("bnf_results_cache"):
        with st.container(border=True):
            bnf_data = st.session_state.bnf_results_cache
            st.markdown(f"#### 📖 BNF Search Results for '{st.session_state.user_input_cache}'")
            
            if bnf_data.get("search_page_summary"):
                with st.expander("View BNF Search Page Summary", expanded=False):
                    st.write(bnf_data["search_page_summary"])
            
            if bnf_data.get("links"):
                st.markdown("**Relevant links from BNF (click to open in a new tab):**")
                for link_info in bnf_data["links"]:
                    st.markdown(f"- [{link_info['title']}]({link_info['url']})")
            elif not bnf_data.get("search_page_summary"):
                 st.info("No specific information or links could be extracted from BNF for this medicine based on the search.")

            if st.session_state.generated_summary: # AI Summary based on search_page_summary
                 st.markdown("---")
                 st.markdown(f"##### ✅ AI-Generated Summary (Based on BNF Search Page Text):")
                 st.write(st.session_state.generated_summary)
            elif bnf_data.get("search_page_summary") and not st.session_state.generated_summary: # If summary text existed but AI failed
                st.warning("Could not generate an AI summary for the BNF search page text.")

    elif st.session_state.generated_summary: # For NHS and Webpage where summary is expected
        with st.container(border=True):
            st.markdown(f"#### ✅ AI-Generated Summary (Based on {st.session_state.source_name} Data):")
            st.write(st.session_state.generated_summary)

            if input_type == "Medical Query":
                similarity_score = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text) # reference_text is NHS text
                st.markdown(f"**Similarity Score (AI Summary vs. {st.session_state.source_name} Data):** `{similarity_score}`/100")
                st.caption(f"This score indicates the semantic similarity between the AI-generated summary and the information retrieved from {st.session_state.source_name}.")
    elif input_type != "Medicine": # If not Medicine and no summary, implies an earlier error
        st.info(f"No AI summary could be generated for {st.session_state.user_input_cache} from {st.session_state.source_name}.")


    if input_type == "Webpage with Medical Claims" and st.session_state.reference_text: 
        st.markdown("---")
        with st.container(border=True):
            st.markdown("#### 🔎 Medical Claims Check on Webpage Content")
            with st.spinner("Analyzing claims on the webpage..."):
                # Use reference_text which is the full webpage content for this input type
                claims_analysis_web = fact_check_post(st.session_state.reference_text, st.session_state.reference_text)
            st.markdown(claims_analysis_web)

    with st.expander("Provide Feedback on this Analysis (Optional)", expanded=False):
        feedback_text = st.text_area("Your feedback on the summary or analysis:", key="feedback_input_area")
        if st.button("Submit Feedback", key="submit_feedback_button"):
            if feedback_text.strip():
                current_input_type_for_feedback = input_type # Capture current selection
                feedback_record = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input": st.session_state.user_input_cache, 
                    "Type": current_input_type_for_feedback, 
                    "Source": st.session_state.source_name,
                    "Summary": st.session_state.generated_summary,
                    "Feedback": feedback_text,
                }
                # Add score only if it's a medical query and the score was relevantly calculated
                if current_input_type_for_feedback == "Medical Query" and st.session_state.reference_text and st.session_state.generated_summary:
                    score_val = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
                    feedback_record["Similarity Score (Summary vs Source)"] = score_val
                
                save_feedback(feedback_record)
            else:
                st.info("Please enter some feedback before submitting.")

    # Social Media Fact-Checking (uses st.session_state.reference_text which is search_summary for BNF)
    # This means for BNF, social media posts will be compared against the search page summary.
    if st.session_state.reference_text and st.session_state.reference_text.strip():
        st.markdown("---")
        st.subheader("🔍 Fact-Check Related Social Media Posts")
        st.caption(f"Social media posts related to '{st.session_state.user_input_cache}' will be checked against the information retrieved from {st.session_state.source_name}.")

        num_posts_to_fetch = st.slider("Number of posts per platform:", min_value=1, max_value=10, value=3, key="num_social_posts")

        tabs = st.tabs(["🐦 Twitter", "💬 Reddit"])

        with tabs[0]: 
            # Twitter Tab Content
            # ... (same as before)
            st.markdown(f"##### Recent Tweets matching '{st.session_state.user_input_cache}'")
            try:
                if 'twitter_client' not in globals() or not isinstance(twitter_client, tweepy.Client):
                     st.warning("Twitter client not initialized. Cannot fetch tweets.")
                else:
                    query = f"{st.session_state.user_input_cache} lang:en -is:retweet -is:quote -is:reply"
                    tweets_response = twitter_client.search_recent_tweets(query=query, max_results=num_posts_to_fetch, tweet_fields=['created_at', 'text', 'public_metrics'])
                    
                    tweets_data = tweets_response.data if tweets_response else None
                    if tweets_data:
                        for i, tweet in enumerate(tweets_data):
                            with st.expander(f"Tweet {i+1}: {tweet.text[:80]}...", expanded= (i==0) ):
                                st.markdown(f"**Tweet {i+1}:**")
                                st.info(tweet.text)
                                with st.spinner(f"Fact-checking Tweet {i+1}..."):
                                    fact_result = fact_check_post(tweet.text, st.session_state.reference_text)
                                    credibility_social = compute_similarity_score(tweet.text, st.session_state.reference_text)
                                st.markdown(f"**Fact-Check Result (vs {st.session_state.source_name}):**")
                                st.write(fact_result)
                                st.markdown(f"**Post Similarity to {st.session_state.source_name} Info:** `{credibility_social}`/100")
                    else:
                        st.info(f"No recent tweets found for '{st.session_state.user_input_cache}'.")
            except tweepy.TweepyException as e:
                logging.error(f"Twitter API error: {e}")
                st.warning(f"Could not retrieve tweets. Twitter API error: {str(e)[:100]}...")
            except Exception as e:
                logging.error(f"Unexpected error fetching/processing tweets: {e}")
                st.warning(f"An error occurred while fetching tweets: {e}")

        with tabs[1]: 
            # Reddit Tab Content
            # ... (same as before)
            st.markdown(f"##### Recent Reddit Posts matching '{st.session_state.user_input_cache}'")
            try:
                if 'reddit' not in globals() or not isinstance(reddit, praw.Reddit):
                     st.warning("Reddit client not initialized. Cannot fetch posts.")
                else:
                    try:
                        subreddit_to_search = reddit.subreddit("AskDocs+medicine+Health+Medical+pharmacy") 
                        reddit_posts = list(subreddit_to_search.search(st.session_state.user_input_cache, limit=num_posts_to_fetch, sort="relevance"))
                    except praw.exceptions.PRAWException as pe:
                        logging.error(f"PRAW specific error fetching Reddit posts: {pe}")
                        st.warning(f"Could not retrieve Reddit posts due to a Reddit API interaction error: {pe}")
                        reddit_posts = []
                    
                    if reddit_posts:
                        for i, post in enumerate(reddit_posts):
                            post_text_full = post.title + ("\n\n" + post.selftext if post.selftext else "")
                            with st.expander(f"Reddit Post {i+1}: {post.title[:80]}... (r/{post.subreddit.display_name})", expanded= (i==0) ):
                                st.markdown(f"**Title:** *{post.title}*")
                                st.caption(f"Subreddit: r/{post.subreddit.display_name} | Score: {post.score} | Comments: {post.num_comments}")
                                if post.selftext:
                                    st.markdown("--- \n**Post Content:**")
                                    st.caption(post.selftext[:1000] + ("..." if len(post.selftext) > 1000 else ""))
                                
                                with st.spinner(f"Fact-checking Reddit Post {i+1}..."):
                                    fact_result = fact_check_post(post_text_full, st.session_state.reference_text)
                                    credibility_social = compute_similarity_score(post_text_full, st.session_state.reference_text)
                                st.markdown(f"**Fact-Check Result (vs {st.session_state.source_name}):**")
                                st.write(fact_result)
                                st.markdown(f"**Post Similarity to {st.session_state.source_name} Info:** `{credibility_social}`/100")
                    else:
                        st.info(f"No relevant Reddit posts found for '{st.session_state.user_input_cache}' in the selected subreddits.")
            except praw.exceptions.PRAWException as e: 
                logging.error(f"PRAW API error (Reddit): {e}")
                st.warning(f"Could not retrieve Reddit posts. Reddit API error: {str(e)[:100]}...")
            except Exception as e: 
                logging.error(f"Unexpected error fetching/processing Reddit posts: {e}")
                st.warning(f"An error occurred while fetching Reddit posts: {e}")

# footer() # Developer credit is in the sidebar
