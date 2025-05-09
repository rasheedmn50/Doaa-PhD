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

# ========== HEADER ==========
st.markdown("""
    <div style='background-color:#012169;padding:10px;text-align:center;'>
        <img src="https://www.bcu.ac.uk/images/default-source/marketing/logos/bcu-logo.svg" width="200"/>
    </div>
""", unsafe_allow_html=True)

# ========== FOOTER ==========
def footer():
    st.markdown("""
    <hr>
    <div style='text-align:center; color: gray;'>
        Developed by <strong>Doaa Al-Turkey</strong>
    </div>
    """, unsafe_allow_html=True)

# ========== API KEYS & SECRETS ==========
# These will be set in Streamlit Cloud's secrets management
# Example structure for your Streamlit secrets:
# OPENAI_API_KEY = "sk-..."
# REDDIT_CLIENT_ID = "..."
# REDDIT_CLIENT_SECRET = "..."
# REDDIT_USER_AGENT = "..."
# TWITTER_BEARER_TOKEN = "..."
# GOOGLE_SHEETS_CREDENTIALS = """
# {
#   "type": "service_account",
#   "project_id": "your-gcp-project-id",
#   ... (rest of your JSON key content)
# }
# """
# GOOGLE_SHEET_NAME = "Your Google Sheet Name" # e.g., "App Feedback"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
    REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
    REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
    TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]

    # Google Sheets Credentials
    GOOGLE_SHEETS_CREDENTIALS_JSON = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    GOOGLE_SHEET_NAME = st.secrets["GOOGLE_SHEET_NAME"]
    
    # Parse the JSON string from secrets
    google_creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
    scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    google_credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scopes)
    gc = gspread.authorize(google_credentials)
    feedback_sheet = gc.open(GOOGLE_SHEET_NAME).sheet1 # Assumes feedback is on the first sheet

except KeyError as e:
    st.error(f"🔴 Missing secret: {e}. Please ensure all API keys and Google Sheets credentials are set in Streamlit Cloud secrets.")
    st.stop()
except Exception as e:
    st.error(f"🔴 Error loading secrets or initializing Google Sheets: {e}")
    logging.error(f"Error loading secrets or initializing Google Sheets: {e}")
    st.stop()


# ========== INIT ==========
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
except Exception as e:
    st.error(f"Error initializing API clients: {e}. Please check your API keys and configurations.")
    logging.error(f"Error initializing API clients: {e}")
    st.stop()

# ========== HELPERS ==========
def fetch_bnf_info(medicine_name):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.service import Service # Added for better control

    base_url = "https://bnf.nice.org.uk"
    search_url = f"{base_url}/search/?q={medicine_name.replace(' ', '%20')}"
    logging.info(f"Constructed BNF search URL: {search_url}")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox") # Crucial for Streamlit Cloud
    options.add_argument("--disable-dev-shm-usage") # Crucial for Streamlit Cloud
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-debugging-port=9222")


    driver = None
    content = ""

    try:
        # For Streamlit Cloud, chromedriver should be available if specified in packages.txt
        # Using Service object for more explicit control
        service = Service() # Assumes chromedriver is in PATH or managed by packages.txt
        driver = webdriver.Chrome(service=service, options=options)
        
        logging.info(f"Attempting to get BNF search page: {search_url}")
        driver.get(search_url)
        
        WebDriverWait(driver, 20).until(
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
            if len(links) >= 3: 
                break
        
        if not links:
            logging.warning("⚠️ No medicine links found on BNF search page for '%s'.", medicine_name)
            st.warning(f"No specific medicine links found on BNF for '{medicine_name}'. The search page content will be used if available, or try a more specific name.")
            page_soup_fallback = BeautifulSoup(driver.page_source, "html.parser")
            main_content_area = page_soup_fallback.find("main")
            if main_content_area:
                 return main_content_area.get_text(separator="\n", strip=True) + "\n"
            return page_soup_fallback.get_text(separator="\n", strip=True) + "\n"

        for link in links:
            logging.info(f"Fetching BNF medicine page: {link}")
            driver.get(link)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_soup = BeautifulSoup(driver.page_source, "html.parser")
            drug_content_area = page_soup.find(id="topic") 
            if drug_content_area:
                 content += drug_content_area.get_text(separator="\n", strip=True) + "\n\n"
            else:
                 content += page_soup.get_text(separator="\n", strip=True) + "\n\n"
            #time.sleep(0.5)

    except TimeoutException:
        logging.error("BNF scraping timed out for URL: %s or subsequent pages.", search_url)
        st.warning("⚠️ Timed out while trying to fetch information from BNF. The site might be slow or content not found as expected.")
    except WebDriverException as e:
        logging.error(f"WebDriver initialization/operation failed for BNF: {e}")
        st.error(f"Error (BNF): Could not operate the web driver: {str(e)[:200]}. This might be a cloud environment issue.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during BNF scraping: {e}")
        st.error(f"An unexpected error occurred while fetching BNF data: {e}")
    finally:
        if driver:
            driver.quit()
            logging.info("WebDriver closed.")
    
    if not content and not links :
        logging.warning("BNF scraping returned no content for '%s'.", medicine_name)
    return content

# ... (fetch_nhs_info, fetch_url_content, generate_response, compute_similarity_score, fact_check_post remain largely the same) ...
# Ensure these functions don't have hardcoded paths or dependencies that won't work in cloud.

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
        return "Reference information was empty or too short to process."

    prompt = (
        f"You are a medical information assistant. Your goal is to provide clear, concise, and layman-friendly information based *only* on the verified medical reference text provided below.\n"
        f"Analyze the following reference information in relation to the topic: '{user_input_topic}'.\n\n"
        f"REFERENCE INFORMATION:\n{truncated_ref_text}\n\n"
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
        # ... (prompt remains the same)
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
        f"VERIFIED REFERENCE INFORMATION:\n\"\"\"{truncated_ref_text}\"\"\"\n\n"
        "Begin your analysis:"
    )
    messages = [{"role": "system", "content": "You are a fact-checking AI specializing in medical claims, comparing a post against provided reference text."},
                {"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.1, max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call for fact-checking failed: {e}")
        return f"Error during fact-checking process: {e}"


def save_feedback(record):
    """Saves a feedback record to the configured Google Sheet."""
    try:
        # The header row in your Google Sheet should match these keys,
        # or you adjust the keys here to match your sheet.
        # Ensure 'Timestamp' is added to the record before calling this.
        row_to_insert = [
            record.get("Timestamp", ""),
            record.get("Input", ""),
            record.get("Type", ""),
            record.get("Source", ""),
            record.get("Summary", ""),
            record.get("Feedback", ""),
            record.get("Similarity Score (Summary vs Source)", "") # Handle if not present
        ]
        feedback_sheet.append_row(row_to_insert)
        logging.info("Feedback saved successfully to Google Sheets.")
        st.success("✅ Feedback recorded. Thank you!")
    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API Error: {e}")
        st.error(f"Could not save feedback to Google Sheets due to API error: {e}. Check share permissions and API enablement.")
    except Exception as e:
        logging.error(f"Failed to save feedback to Google Sheets: {e}")
        st.error(f"Could not save feedback: {e}")

# ========== STREAMLIT UI ==========
# ... (The Streamlit UI layout remains the same, but the `save_feedback` call will now use the Google Sheets version) ...
# Make sure the `feedback_record` dictionary in the UI part has a "Timestamp"
# Example adjustment in the UI section for feedback submission:

st.title("🧠 Validate Medicine Information")

input_type = st.radio(
    "What are you analyzing?",
    ["Medicine", "Medical Query", "Webpage with Medical Claims"],
    horizontal=True, key="input_type_radio"
)

user_input_label = "Enter Medicine Name (e.g., Paracetamol):"
if input_type == "Medical Query": user_input_label = "Enter Medical Query (e.g., symptoms of flu):"
elif input_type == "Webpage with Medical Claims": user_input_label = "Enter Webpage URL with Medical Claims (e.g., https://example.com/article):"

user_input = st.text_input(user_input_label, key="user_main_input")

if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'reference_text' not in st.session_state: st.session_state.reference_text = ""
if 'generated_summary' not in st.session_state: st.session_state.generated_summary = ""
if 'source_name' not in st.session_state: st.session_state.source_name = ""


if st.button("Analyze", key="analyze_button"):
    st.session_state.analysis_done = False 
    st.session_state.reference_text = ""
    st.session_state.generated_summary = ""
    st.session_state.source_name = ""

    if not user_input.strip():
        st.warning("Please enter a valid input.")
    else:
        with st.spinner(f"Fetching and analyzing information for '{user_input[:50]}...'"):
            reference_text = ""
            source_name = ""

            if input_type == "Medicine":
                reference_text = fetch_bnf_info(user_input)
                source_name = "BNF"
            elif input_type == "Medical Query":
                reference_text = fetch_nhs_info(user_input)
                source_name = "NHS"
            else: 
                if not (user_input.startswith("http://") or user_input.startswith("https://")):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    reference_text = fetch_url_content(user_input)
                    source_name = "Webpage"
            
            st.session_state.reference_text = reference_text
            st.session_state.source_name = source_name

            if not reference_text or not reference_text.strip():
                st.error(f"❌ Could not retrieve reference information from {source_name} for '{user_input}'. Please check the input or try again later.")
            else:
                st.session_state.generated_summary = generate_response(user_input, reference_text)
                st.session_state.analysis_done = True


if st.session_state.analysis_done:
    st.markdown(f"### ✅ AI-Generated Summary (Based on {st.session_state.source_name} Data):")
    st.write(st.session_state.generated_summary)

    if input_type == "Medical Query":
        similarity_score = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
        st.markdown(f"**Similarity Score (AI Summary vs. {st.session_state.source_name} Data):** {similarity_score}/100")
        st.caption(f"This score indicates the semantic similarity between the AI-generated summary and the information retrieved from {st.session_state.source_name}.")
    
    if input_type == "Webpage with Medical Claims":
        st.markdown("---")
        st.markdown("### 🔎 Medical Claims Check on Webpage Content")
        with st.spinner("Analyzing claims on the webpage..."):
            claims_analysis_web = fact_check_post(st.session_state.reference_text, st.session_state.reference_text)
        st.markdown(claims_analysis_web)

    with st.expander("Provide Feedback (Optional)", expanded=False):
        feedback_text = st.text_area("Your feedback on the summary or analysis:", key="feedback_input")
        if st.button("Submit Feedback", key="submit_feedback_button"):
            if feedback_text.strip():
                feedback_record = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), # Added Timestamp
                    "Input": user_input, 
                    "Type": input_type, 
                    "Source": st.session_state.source_name,
                    "Summary": st.session_state.generated_summary, 
                    "Feedback": feedback_text,
                    # "Similarity Score (Summary vs Source)" will be added if applicable
                }
                if input_type == "Medical Query": # Only add this score if it was calculated
                    score_val = compute_similarity_score(st.session_state.generated_summary, st.session_state.reference_text)
                    feedback_record["Similarity Score (Summary vs Source)"] = score_val
                
                save_feedback(feedback_record) # This now calls the Google Sheets version
                # No st.success here, save_feedback shows it
            else:
                st.info("Please enter some feedback before submitting.")

    if st.session_state.reference_text and st.session_state.reference_text.strip():
        st.markdown("---")
        st.markdown("## 🔍 Fact-Check Related Social Media Posts")
        st.caption(f"Recent social media posts related to '{user_input}' will be checked against the information retrieved from {st.session_state.source_name}.")

        tabs = st.tabs(["🐦 Twitter", "💬 Reddit"])

        with tabs[0]: 
            st.markdown(f"##### Recent Tweets matching '{user_input}'")
            try:
                if 'twitter_client' not in globals() or not isinstance(twitter_client, tweepy.Client):
                     st.warning("Twitter client not initialized. Cannot fetch tweets.")
                else:
                    query = f"{user_input} lang:en -is:retweet -is:quote -is:reply"
                    tweets_response = twitter_client.search_recent_tweets(query=query, max_results=5, tweet_fields=['created_at', 'text', 'public_metrics'])
                    
                    tweets_data = tweets_response.data if tweets_response else None # Ensure tweets_data is None if no response
                    if tweets_data: # Check if tweets_data is not None and has content
                        for i, tweet in enumerate(tweets_data):
                            post_text = tweet.text
                            st.markdown(f"**Tweet {i+1}:**")
                            st.info(post_text)
                            with st.spinner(f"Fact-checking Tweet {i+1}..."):
                                fact_result = fact_check_post(post_text, st.session_state.reference_text)
                                credibility_social = compute_similarity_score(post_text, st.session_state.reference_text)
                            st.markdown(f"**Fact-Check Result (vs {st.session_state.source_name}):**")
                            st.write(fact_result)
                            st.markdown(f"**Post Similarity to {st.session_state.source_name} Info:** {credibility_social}/100")
                            st.markdown("---")
                    else:
                        st.info(f"No recent tweets found for '{user_input}'.")
            except tweepy.TweepyException as e:
                logging.error(f"Twitter API error: {e}")
                st.warning(f"Could not retrieve tweets. Twitter API error: {str(e)[:100]}...")
            except Exception as e:
                logging.error(f"Unexpected error fetching/processing tweets: {e}")
                st.warning(f"An error occurred while fetching tweets: {e}")

        with tabs[1]: 
            st.markdown(f"##### Recent Reddit Posts matching '{user_input}'")
            try:
                if 'reddit' not in globals() or not isinstance(reddit, praw.Reddit):
                     st.warning("Reddit client not initialized. Cannot fetch posts.")
                else:
                    try:
                        subreddit_to_search = reddit.subreddit("AskDocs+medicine+Health") 
                        reddit_posts = list(subreddit_to_search.search(user_input, limit=5, sort="relevance"))
                    except praw.exceptions.PRAWException as pe:
                        logging.error(f"PRAW specific error fetching Reddit posts: {pe}")
                        st.warning(f"Could not retrieve Reddit posts due to a Reddit API interaction error: {pe}")
                        reddit_posts = []
                    
                    if reddit_posts:
                        for i, post in enumerate(reddit_posts):
                            post_text = post.title + ("\n\n" + post.selftext if post.selftext else "")
                            st.markdown(f"**Reddit Post {i+1} (From r/{post.subreddit.display_name}):** *{post.title}*")
                            with st.expander("View Post Content & Fact-Check"):
                                st.caption(post.selftext[:500] + "..." if len(post.selftext) > 500 else post.selftext)
                                with st.spinner(f"Fact-checking Reddit Post {i+1}..."):
                                    fact_result = fact_check_post(post_text, st.session_state.reference_text)
                                    credibility_social = compute_similarity_score(post_text, st.session_state.reference_text)
                                st.markdown(f"**Fact-Check Result (vs {st.session_state.source_name}):**")
                                st.write(fact_result)
                                st.markdown(f"**Post Similarity to {st.session_state.source_name} Info:** {credibility_social}/100")
                            st.markdown("---")
                    else:
                        st.info(f"No recent Reddit posts found for '{user_input}' in the selected subreddits.")
            except praw.exceptions.PRAWException as e: 
                logging.error(f"PRAW API error (Reddit): {e}")
                st.warning(f"Could not retrieve Reddit posts. Reddit API error: {str(e)[:100]}...")
            except Exception as e: 
                logging.error(f"Unexpected error fetching/processing Reddit posts: {e}")
                st.warning(f"An error occurred while fetching Reddit posts: {e}")

footer()
