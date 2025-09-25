# main.py - Refactored AI News Dashboard Data Processing

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import json
import time
import spacy
import openai
from dotenv import load_dotenv
import os
from geopy.geocoders import Nominatim
from datetime import datetime

# --------------------------
# CONFIGURATION
# --------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")
openai.api_key = OPENAI_API_KEY


RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/cnn_world.rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.wired.com/feed/rss",
    "https://www.wired.com/feed/rss",
    "https://www.bellingcat.com/feed/",
    "https://bellingcat.libsyn.com/rss",
    "https://balkaninsight.com/feed/",
    "https://news.ycombinator.com/rss",
    "https://www.cshub.com/rss/categories/attacks",
    "https://www.crisisgroup.org/rss",
    "https://intelnews.org/feed/",


    # Add more feeds as desired
]

DB_PATH = "articles.db"
MODEL_NAME = "gpt-3.5-turbo-instruct"

nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="osint_news_module")
GEOCODE_CACHE = {}
# --------------------------
# LOAD GEOCODE CACHE
# --------------------------
import json

try:
    with open("geocode_cache.json", "r") as f:
        GEOCODE_CACHE = json.load(f)
except FileNotFoundError:
    GEOCODE_CACHE = {}

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def safe_load_json(x):
    if pd.isnull(x):
        return {}
    try:
        return json.loads(x)
    except:
        return {}

def get_dominant_category_safe(cat_scores):
    if not cat_scores or cat_scores == {}:
        return 'Unknown'
    try:
        return max(cat_scores, key=cat_scores.get)
    except:
        return 'Unknown'

def geocode_location(location):
    if not location:
        return None, None
    if location in GEOCODE_CACHE:
        return GEOCODE_CACHE[location]
    try:
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            GEOCODE_CACHE[location] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except:
        pass
    GEOCODE_CACHE[location] = (None, None)
    return None, None

# --------------------------
# GET EXISTING LINKS FROM DB
# --------------------------
def get_existing_links(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT link FROM articles")
    links = set([row[0] for row in c.fetchall()])
    conn.close()
    return links

# --------------------------
# STEP 1: SCRAPE NEWS
# --------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_feed(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return []
        soup = BeautifulSoup(response.content, "lxml-xml")
        items = soup.find_all('item')
        articles = []

        for item in items:
            # Only include items that have both title and link
            if item.title and item.link:
                articles.append({
                    "title": item.title.text,
                    "link": item.link.text,
                    "pub_date": getattr(item.pubDate, "text", ""),
                    "description": getattr(item.description, "text", "")
                })
        return articles
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def scrape_rss(feeds, max_per_feed=1):
    """Parallelized RSS scraping using threads, limited to max_per_feed per feed."""
    articles = []

    def fetch_limited(url):
        feed_articles = fetch_feed(url)
        return feed_articles[:max_per_feed]  # take only first n articles

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_limited, url): url for url in feeds}
        for future in as_completed(future_to_url):
            result = future.result()
            articles.extend(result)

    print(f"Scraped {len(articles)} articles (max {max_per_feed} per feed).")
    return articles

# --------------------------
# STEP 2: AI SUMMARIZATION + THREAT DETECTION (Updated for OpenAI >=1.0.0)
# --------------------------
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def categorize_and_score(article_text):
    """
    Uses OpenAI GPT-4 to summarize an article, assign category scores,
    and compute overall risk score. Returns a JSON dictionary with:
    - summary (str)
    - categories (dict)
    - overall_risk_score (float)
    """
    prompt = f"""
    You are an AI analyst. Read the following news article and do the following:
    1. Summarize it in 2-3 sentences.
    2. Identify threat categories (cyber, military, political, space/satellite) with confidence scores 0-1.
    3. Calculate overall risk score 0-1.
    Return JSON with keys: summary, categories, overall_risk_score.
    Article: {article_text}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        # Extract content from response
        content = response.choices[0].message.content
        # Convert string to dictionary
        return json.loads(content)
    except Exception as e:
        print(f"AI analysis failed: {e}")
        # Fallback if AI fails
        return {"summary": article_text[:150]+"...", "categories": {}, "overall_risk_score": 0}
# --------------------------
# STEP 3: LOCATION EXTRACTION
# --------------------------
def extract_locations(text):
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]})

# --------------------------
# STEP 4: DATABASE OPERATIONS
# --------------------------
def setup_db(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            pub_date TEXT,
            description TEXT,
            summary TEXT,
            category_scores TEXT,
            dominant_category TEXT,
            locations TEXT,
            lat REAL,
            lon REAL,
            overall_risk_score REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_articles_to_db(articles, path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    for article in articles:
        lat, lon = None, None
        if article['locations']:
            # Take first location for mapping purposes
            lat, lon = geocode_location(article['locations'][0])
        c.execute('''
            INSERT OR REPLACE INTO articles
            (title, link, pub_date, description, summary, category_scores, dominant_category, locations, lat, lon, overall_risk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['title'],
            article['link'],
            article['pub_date'],
            article['description'],
            article['summary'],
            json.dumps(article['category_scores']),
            get_dominant_category_safe(article['category_scores']),
            json.dumps(article['locations']),
            lat,
            lon,
            article['overall_risk_score']
        ))
    conn.commit()
    conn.close()
    print(f"Saved {len(articles)} articles to database.")

# --------------------------
# MAIN EXECUTION
# --------------------------
def main():
    setup_db(DB_PATH)

    # Load geocode cache at the start (already handled at top of file)
    global GEOCODE_CACHE
    try:
        with open("geocode_cache.json", "r") as f:
            GEOCODE_CACHE = json.load(f)
    except FileNotFoundError:
        GEOCODE_CACHE = {}

    # Scrape articles from RSS feeds, limit 1 article per feed to avoid API overuse
    articles = scrape_rss(RSS_FEEDS, max_per_feed=1)

    # Get already processed links from the database
    existing_links = get_existing_links(DB_PATH)

    for i, article in enumerate(articles):
        if article['link'] in existing_links:
            print(f"Skipping already processed article: {article['title']}")
            continue

        print(f"Processing article {i+1}/{len(articles)}: {article['title']}")

        # AI summarization and threat detection
        res = categorize_and_score(article['description'])
        article['summary'] = res['summary']
        article['category_scores'] = res['categories']
        article['overall_risk_score'] = res['overall_risk_score']

        # Extract locations
        article['locations'] = extract_locations(article['description'])

        time.sleep(1)  # optional delay to prevent API overload

    # Save all new articles to the database
    save_articles_to_db(articles, DB_PATH)

    # Save geocode cache for future runs
    with open("geocode_cache.json", "w") as f:
        json.dump(GEOCODE_CACHE, f)

if __name__ == "__main__":
    main()