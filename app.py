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
# Example structure for your Streamlit secrets (in your app's advanced settings on Streamlit Cloud):
# OPENAI_API_KEY = "sk-..."
# REDDIT_CLIENT_ID = "..."
# REDDIT_CLIENT_SECRET = "..."
# REDDIT_USER_AGENT = "YourAppName by /u/YourRedditUsername"
# TWITTER_BEARER_TOKEN = "..."
# GOOGLE_SHEETS_CREDENTIALS = """
# {
#   "type": "service_account",
#   "project_id": "your-gcp-project-id",
#   ... (rest of your JSON key content as a multi-line string)
# }
# """
# GOOGLE_SHEET_NAME = "Your Google Sheet File Name" # e.g., "App Feedback"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
    REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
    REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
    TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]

    # Google Sheets Credentials
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
    model = SentenceTransformer('all-MiniLM-L6-v2') # Or your preferred model
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"Error initializing API clients (OpenAI, SentenceTransformer, PRAW, Tweepy): {e}. Please check API keys and configurations if applicable.")
    logging.error(f"Error initializing API clients: {e}")
    st.stop()

# ========== HELPER FUNCTIONS ==========
def fetch_bnf_info(medicine_name):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.service import Service

    base_url = "https://bnf.nice.org.uk"
    search_url = f"{base_url}/search/?q={medicine_name.replace(' ', '%20')}"
    logging.info(f"Constructed BNF search URL: {search_url}")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36")

    driver = None
    content = ""

    try:
        service = Service() 
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(45)

        logging.info(f"Attempting to get BNF search page: {search_url}")
        driver.get(search_url)
        
        WebDriverWait(driver, 25).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "card__header"))
        )
        logging.info("BNF search page loaded and card headers found.")
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        headers = soup.find_all("header", class_="card__header")
        links = []
        
        for header in headers:
            a_tag = header.find("a", href=True)
            if a_tag and a_tag["href"].startswith("/"):
                links.append(base_url + a_tag["href"])
            if len(links) >= 2: 
                break
        
        if not links:
            logging.warning("⚠️ No medicine links found on BNF search page for '%s'.", medicine_name)
            page_soup_fallback = BeautifulSoup(driver.page_source, "html.parser")
            main_content_area = page_soup_fallback.find("main", id="maincontent") or \
                                page_soup_fallback.find("div", class_="search-results") or \
                                page_soup_fallback.find("main")
            if main_content_area:
                st.warning(f"No specific medicine links found on BNF for '{medicine_name}'. Using content from the search results page.")
                return main_content_area.get_text(separator="\n", strip=True) + "\n"
            else:
                st.warning(f"No specific medicine links or primary content found on BNF search page for '{medicine_name}'.")
                return page_soup_fallback.get_text(separator="\n", strip=True) + "\n"

        for link_idx, link in enumerate(links):
            logging.info(f"Fetching BNF medicine page {link_idx+1}/{len(links)}: {link}")
            try:
                driver.get(link)
                WebDriverWait(driver, 40).until(
                    EC.visibility_of_element_located((By.ID, "topic"))
                )
                page_soup = BeautifulSoup(driver.page_source, "html.parser")
                drug_content_area = page_soup.find(id="topic")
                if drug_content_area:
                    content += drug_content_area.get_text(separator="\n", strip=True) + "\n\n---\n\n"
                else:
                    main_body_content = page_soup.find("body")
                    if main_body_content:
                        content += main_body_content.get_text(separator="\n", strip=True) + "\n\n---\n\n"
                # time.sleep(0.2) 
            except TimeoutException:
                logging.error(f"BNF: Timed out waiting for content on drug page: {link}")
                st.warning(f"⚠️ Timed out fetching full details for one of the BNF links ({link}). Partial information might be available if other links succeeded.")
                continue 
            except WebDriverException as page_e:
                logging.error(f"BNF: WebDriverException on drug page {link}: {page_e}")
                st.warning(f"⚠️ Error loading one of the BNF links ({link}).")
                continue

    except TimeoutException as e:
        logging.error(f"BNF: Overall timeout during operation for {search_url} or initial page. Exception: {e}")
        st.warning(f"⚠️ Timed out connecting to or loading initial BNF search page for '{medicine_name}'. The site might be very busy or inaccessible.")
    except WebDriverException as e:
        logging.error(f"BNF: WebDriver initialization/operation failed: {e}")
        st.error(f"Error (BNF): Could not operate the web driver: {str(e)[:200]}. This might be a cloud environment issue or BNF site structure change.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during BNF scraping: {e}")
        st.error(f"An unexpected error occurred while fetching BNF data: {e}")
    finally:
        if driver:
            driver.quit()
            logging.info("WebDriver closed for BNF.")
    
    if not content:
        logging.warning("BNF scraping returned no usable content for '%s'.", medicine_name)
        st.warning(f"Could not retrieve detailed information from BNF for '{medicine_name}'. This could be due to timeouts, no matching drug found, or site changes.")
    return content.strip().removesuffix("---\n\n")

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
                    time.sleep(0.5) # Be respectful
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
        
        text = re.sub(r'\n\s*\n', '\n\n', text) # Consolidate multiple newlines
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
        # Consider encoding shorter text first if lengths are very different, though for sentence transformers it's usually fine.
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        similarity = util.cos_sim(emb_a, emb_b).item()
        score = round(max(0, similarity * 100), 2) # Ensure score is not negative due to floating point issues
        return score
    except Exception as e:
        logging.error(f"Error computing similarity score: {e}")
        return 0.0

def fact_check_post(post_text, reference_text):
    max_ref_length = 8000 # Max length for reference text in fact-checking prompt
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
            record.get("Timestamp", ""),
            record.get("Input", ""),
            record.get("Type", ""),
            record.get("Source", ""),
            record.get("Summary", ""),
            record.get("Feedback", ""),
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
    # You can host the image on GitHub in your repository and use the raw link,
    # or use a publicly accessible URL if the bcu.ac.uk one is stable.
    # For GitHub, find the image in your repo, click on it, then click "Download" or "Raw" to get the direct link.
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
    'source_name': "", 'user_input_cache': ""
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.header("Analyze Medical Information")

input_type = st.radio(
    "What are you analyzing?",
    ["Medicine", "Medical Query", "Webpage with Medical Claims"],
    horizontal=True, key="input_type_radio",
    captions = ["e.g., Paracetamol", "e.g., symptoms of flu", "e.g., URL of an article"]
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
        for key in ['analysis_done', 'reference_text', 'generated_summary', 'source_name']:
            st.session_state[key] = default_session_state[key]
        # st.session_state.user_input_cache = "" # Uncomment to also clear the input field
        st.rerun()

if analyze_button_pressed:
    st.session_state.analysis_done = False
    st.session_state.reference_text = ""
    st.session_state.generated_summary = ""
    st.session_state.source_name = ""
    st.session_state.user_input_cache = user_input

    if not user_input.strip():
        st.warning("Please enter a valid input.")
    else:
        with st.status(f"Analyzing '{user_input[:30]}...' ({input_type})", expanded=True) as status_ui:
            reference_text = ""
            source_name = ""
            
            status_ui.update(label=f"Fetching information from source ({input_type})... Please wait.")
            if input_type == "Medicine":
                reference_text = fetch_bnf_info(user_input)
                source_name = "BNF"
            elif input_type == "Medical Query":
                reference_text = fetch_nhs_info(user_input)
                source_name = "NHS"
            else: 
                if not (user_input.startswith("http://") or user_input.startswith("https://")):
                    st.error("Please enter a valid URL starting with http:// or https://")
                    status_ui.update(label="Invalid URL provided.", state="error", expanded=False)
                    # Set reference_text to None or empty to prevent further processing if URL is invalid
                    reference_text = None 
                else:
                    reference_text = fetch_url_content(user_input)
                    source_name = "Webpage"
            
            # Check if reference_text is None (due to invalid URL) or empty
            if reference_text is None: # Explicit check for the invalid URL case
                 pass # Error already shown, status updated.
            elif not reference_text or not reference_text.strip():
                st.error(f"❌ Could not retrieve reference information from {source_name} for '{user_input}'. Please check the input or try again later.")
                status_ui.update(label=f"Failed to get data from {source_name}.", state="error", expanded=False)
            else:
                st.session_state.reference_text = reference_text
                st.session_state.source_name = source_name
                status_ui.update(label=f"Information retrieved from {source_name}. Generating AI summary...")
                st.session_state.generated_summary = generate_response(user_input, reference_text)
                st.session_state.analysis_done = True
                status_ui.update(label="Analysis complete!", state="complete", expanded=False)

if st.session_state.analysis_done:
    st.markdown("---")
    st.subheader(f"Analysis Results for: \"{st.session_state.user_input_cache}\"")

    with st.container(border=True):
        st.markdown(f"#### ✅ AI-Generated Summary (Based on {st.session_state.source_name} Data):")
        st.write(st.session_state.generated_summary)

        if input_type == "Medical Query":
            similarity_score = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
            st.markdown(f"**Similarity Score (AI Summary vs. {st.session_state.source_name} Data):** `{similarity_score}`/100")
            st.caption(f"This score indicates the semantic similarity between the AI-generated summary and the information retrieved from {st.session_state.source_name}.")
    
    if input_type == "Webpage with Medical Claims" and st.session_state.reference_text : # Ensure ref text exists
        st.markdown("---")
        with st.container(border=True):
            st.markdown("#### 🔎 Medical Claims Check on Webpage Content")
            with st.spinner("Analyzing claims on the webpage..."):
                claims_analysis_web = fact_check_post(st.session_state.reference_text, st.session_state.reference_text)
            st.markdown(claims_analysis_web)

    with st.expander("Provide Feedback on this Analysis (Optional)", expanded=False):
        feedback_text = st.text_area("Your feedback on the summary or analysis:", key="feedback_input_area")
        if st.button("Submit Feedback", key="submit_feedback_button"):
            if feedback_text.strip():
                feedback_record = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input": st.session_state.user_input_cache, 
                    "Type": input_type, # This will be the input_type active when feedback is submitted
                    "Source": st.session_state.source_name,
                    "Summary": st.session_state.generated_summary,
                    "Feedback": feedback_text,
                }
                if input_type == "Medical Query":
                    score_val = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
                    feedback_record["Similarity Score (Summary vs Source)"] = score_val
                
                save_feedback(feedback_record)
            else:
                st.info("Please enter some feedback before submitting.")

    if st.session_state.reference_text and st.session_state.reference_text.strip():
        st.markdown("---")
        st.subheader("🔍 Fact-Check Related Social Media Posts")
        st.caption(f"Recent social media posts related to '{st.session_state.user_input_cache}' will be checked against the information retrieved from {st.session_state.source_name}.")

        num_posts_to_fetch = st.slider("Number of posts per platform:", min_value=1, max_value=10, value=3, key="num_social_posts")

        tabs = st.tabs(["🐦 Twitter", "💬 Reddit"])

        with tabs[0]: 
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
            st.markdown(f"##### Recent Reddit Posts matching '{st.session_state.user_input_cache}'")
            try:
                if 'reddit' not in globals() or not isinstance(reddit, praw.Reddit):
                     st.warning("Reddit client not initialized. Cannot fetch posts.")
                else:
                    try:
                        # Consider allowing user to specify subreddits or using a broader search
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

# The footer() function is not called here as developer credit is in the sidebar.
# If you want the footer at the bottom of the main page as well, uncomment the next line:
# footer()
