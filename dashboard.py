import sqlite3
import pandas as pd
import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from geopy.geocoders import Nominatim
import time

# --------------------------
# LOAD DATA
# --------------------------
db_path = "articles.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()

# Safe conversion of JSON strings
def safe_load_json(x):
    if pd.isnull(x):
        return {}
    try:
        return json.loads(x)
    except:
        return {}

df['category_scores'] = df['category_scores'].apply(safe_load_json)
df['locations'] = df['locations'].apply(lambda x: json.loads(x) if pd.notnull(x) else [])
df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

# Compute dominant_category safely
def get_dominant_category_safe(cat_scores):
    if not cat_scores or cat_scores == {}:
        return 'Unknown'
    try:
        return max(cat_scores, key=cat_scores.get)
    except:
        return 'Unknown'

df['dominant_category'] = df['category_scores'].apply(get_dominant_category_safe)

# --------------------------
# GEOCODING LOCATIONS
# --------------------------
geolocator = Nominatim(user_agent="osint_news_dashboard")
geocode_cache = {}

# Explode locations for mapping
map_df = df.explode('locations').dropna(subset=['locations'])

latitudes = []
longitudes = []

for loc in map_df['locations']:
    if loc in geocode_cache:
        lat, lon = geocode_cache[loc]
    else:
        try:
            geo = geolocator.geocode(loc, timeout=10)
            if geo:
                lat, lon = geo.latitude, geo.longitude
            else:
                lat, lon = None, None
        except:
            lat, lon = None, None
        geocode_cache[loc] = (lat, lon)
        time.sleep(1)
    latitudes.append(lat)
    longitudes.append(lon)

map_df['lat'] = latitudes
map_df['lon'] = longitudes
map_df = map_df.dropna(subset=['lat', 'lon'])

# --------------------------
# DASH APP
# --------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Filters"),
            html.Label("Select Categories:"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': cat, 'value': cat} for cat in df['dominant_category'].unique() if cat != 'Unknown'],
                multi=True,
                placeholder="Filter by category"
            ),
            html.Label("Minimum Risk Score:"),
            dcc.Slider(
                id='risk-slider',
                min=0,
                max=1,
                step=0.05,
                value=0,
                marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'}
            )
        ], width=3),
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label='Timeline', children=[
                    dcc.Graph(id='timeline-graph')
                ]),
                dcc.Tab(label='Global Threat Map', children=[
                    dcc.Graph(id='map-graph')
                ]),
                dcc.Tab(label='News Feed', children=[
                    dbc.Table.from_dataframe(df[['pub_date','title','dominant_category','overall_risk_score']], striped=True, bordered=True, hover=True, responsive=True)
                ])
            ])
        ], width=9)
    ])
], fluid=True)

# --------------------------
# CALLBACKS
# --------------------------
@app.callback(
    [Output('timeline-graph', 'figure'),
     Output('map-graph', 'figure')],
    [Input('category-dropdown', 'value'),
     Input('risk-slider', 'value')]
)
def update_graphs(selected_categories, min_risk):
    filtered_df = df[df['overall_risk_score'] >= min_risk]
    filtered_map_df = map_df[map_df['overall_risk_score'] >= min_risk]

    if selected_categories:
        filtered_df = filtered_df[filtered_df['dominant_category'].isin(selected_categories)]
        filtered_map_df = filtered_map_df[filtered_map_df['dominant_category'].isin(selected_categories)]

    # Timeline
    timeline_df = filtered_df.copy()
    timeline_df['dominant_category'] = timeline_df['category_scores'].apply(get_dominant_category_safe)
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

    # Map
    filtered_map_df['dominant_category'] = filtered_map_df['category_scores'].apply(get_dominant_category_safe)
    fig_map = px.scatter_mapbox(
        filtered_map_df,
        lat='lat',
        lon='lon',
        color='dominant_category',
        size='overall_risk_score',
        hover_name='title',
        hover_data=['summary','category_scores','locations'],
        zoom=1,
        mapbox_style="open-street-map",
        title='Global Threat Map from News'
    )

    return fig_timeline, fig_map

# --------------------------
# RUN DASH APP
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)