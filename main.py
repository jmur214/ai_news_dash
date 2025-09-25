# main.py
# Full OSINT news processing script (steps 1-10)

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import sqlite3
import json
from datetime import datetime
import time
import spacy
import openai
import plotly.express as px
from dateutil import parser
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file or environment variables.")

# Assign to OpenAI
openai.api_key = OPENAI_API_KEY

# --------------------------
# CONFIGURATION
# --------------------------


# RSS feed URL (can add more later)
rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"

# SQLite database
db_path = "articles.db"

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# --------------------------
# STEP 1: SCRAPE NEWS
# --------------------------
response = requests.get(rss_url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch RSS feed, status code: {response.status_code}")

# Use lxml-xml parser
soup = BeautifulSoup(response.content, "lxml-xml")
items = soup.find_all('item')

articles = []
for item in items:
    title = item.title.text
    link = item.link.text
    pub_date = item.pubDate.text
    description = item.description.text
    articles.append({
        "title": title,
        "link": link,
        "pub_date": pub_date,
        "description": description
    })

print(f"Scraped {len(articles)} articles.")

# --------------------------
# STEP 2: AI SUMMARIZATION + THREAT DETECTION
# --------------------------
def categorize_and_score(article_text):
    prompt = f"""
    You are an AI analyst. Read the following news article and do the following:
    1. Summarize it in 2-3 sentences.
    2. Identify all threat categories mentioned. Categories: cyber, military, political, space/satellite.
       For each category, assign a confidence score from 0 (not present) to 1 (highly relevant).
    3. Calculate an overall risk score from 0 to 1 for the event, based on severity and potential impact.
    Return JSON with keys: "summary", "categories" (dict of category: confidence), "overall_risk_score".
    Article: {article_text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response['choices'][0]['message']['content']
        return json.loads(content)
    except:
        # fallback
        return {"summary": article_text[:150]+"...", "categories": {}, "overall_risk_score": 0}

for i, article in enumerate(articles):
    print(f"Processing article {i+1}/{len(articles)}: {article['title']}")
    res = categorize_and_score(article['description'])
    article['summary'] = res['summary']
    article['category_scores'] = res['categories']
    article['overall_risk_score'] = res['overall_risk_score']
    time.sleep(1)  # avoid hitting OpenAI rate limits

# --------------------------
# STEP 3: LOCATION EXTRACTION
# --------------------------
def extract_locations(text):
    doc = nlp(text)
    locations = list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]})
    return locations

for article in articles:
    article['locations'] = extract_locations(article['description'])

# --------------------------
# STEP 4: STORE IN SQLITE
# --------------------------
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create table if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    link TEXT UNIQUE,
    pub_date TEXT,
    description TEXT,
    summary TEXT,
    category_scores TEXT,
    locations TEXT,
    overall_risk_score REAL
)
''')

# Insert (new) articles 
for article in articles:
    c.execute('''
        INSERT OR IGNORE INTO articles 
        (title, link, pub_date, description, summary, category_scores, locations, overall_risk_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        article['title'],
        article['link'],
        article['pub_date'],
        article['description'],
        article['summary'],
        json.dumps(article['category_scores']),
        json.dumps(article['locations']),
        article['overall_risk_score']
    ))

conn.commit()
conn.close()
print(f"Saved {len(articles)} articles to {db_path}")

# --------------------------
# STEP 5: TIMELINE VISUALIZATION
# --------------------------
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()

df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

# Explode categories for plotting
timeline_df = df.explode('category_scores')
timeline_df['category_scores'] = timeline_df['category_scores'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
timeline_df['dominant_category'] = timeline_df['category_scores'].apply(lambda x: max(x, key=x.get) if x else 'Unknown')

fig_timeline = px.timeline(
    timeline_df,
    x_start='pub_date',
    x_end='pub_date',
    y='dominant_category',
    color='dominant_category',
    hover_data=['title', 'summary', 'locations', 'overall_risk_score'],
    title='Timeline of Flagged Events by Category'
)
fig_timeline.update_yaxes(categoryorder='total ascending')
fig_timeline.show()

# --------------------------
# STEP 6: WORLD MAP VISUALIZATION
# --------------------------
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="osint_news_module")

# Explode locations for mapping
map_df = df.explode('locations').dropna(subset=['locations'])

# Add dominant_category column so Plotly can color by it
map_df['category_scores'] = map_df['category_scores'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
map_df['dominant_category'] = map_df['category_scores'].apply(
    lambda x: max(x, key=x.get) if x else 'Unknown'
)

# Ensure lat/lon exist if using geocoding
if 'lat' not in map_df.columns:
    map_df['lat'] = None
if 'lon' not in map_df.columns:
    map_df['lon'] = None
# Geocode locations to get lat/lon
def geocode_location(location):
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
    except:
        return None, None
    return None, None

# Add lat/lon columns if not present
if 'lat' not in map_df.columns or 'lon' not in map_df.columns:
    latitudes = []
    longitudes = []
    for loc in map_df['locations']:
        lat, lon = geocode_location(loc)
        latitudes.append(lat)
        longitudes.append(lon)
    map_df['lat'] = latitudes
    map_df['lon'] = longitudes

# Drop rows where geocoding failed
map_df = map_df.dropna(subset=['lat', 'lon'])

# Now create the map
fig_map = px.scatter_geo(
    map_df,
    lat='lat',
    lon='lon',
    color='dominant_category',  # fixed issue here
    size='overall_risk_score',
    hover_name='title',
    hover_data=['summary', 'category_scores', 'locations'],
    projection='natural earth',
    title='Global Threat Map from News'
)

fig_map.show()
