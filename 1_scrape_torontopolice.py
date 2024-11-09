""" 
scrape_torontopolice.py
Goal: Scraping from Crime Prevention Articles (and other sources), using FireCrawl.
"""

# ------ Package installation ------

# Packages to handle pip installs
import os
import sys
import subprocess
import pkg_resources

from dotenv import load_dotenv
import json

# Import Statements for writing/reading with Google Sheets API
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from urllib.parse import urlparse, quote
import chardet
from bs4 import BeautifulSoup
import html2text
import PyPDF2
import time
import re

# FireCrawl App:
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Variables:
JSON_FILE = "torontopublicsafetycorpus.json"
crawl_results = [] # store in json format


# Some keys and JSON files for API Keys are hidden in this repo.


# ------ Part 1: Check Package Requirements ------

def check_and_install_requirements():
    requirements_path = 'requirements.txt'
    with open(requirements_path, 'r') as f:
        required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = set(required_packages) - installed
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path], stdout=subprocess.DEVNULL)
        print("Installation complete.")

check_and_install_requirements()


# ------ Connect to Google Sheets again ------

# Configuration
SPREADSHEET_ID = '1PEVuqlUrvVoJJUflYGq4gN6itj9WZzGB-UHYnsIusng'  # Updated with correct ID
RANGE_NAME = 'Sheet1!C:C'
SERVICE_ACCOUNT_FILE = 'torontopolicellmbot-9644404c0bdf.json'

# Initialize Google Sheets API
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, 
    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
)
service = build('sheets', 'v4', credentials=credentials)

def get_urls_from_sheet():
    try:
        # Direct API call without using sheet variable
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME
        ).execute()
        values = result.get('values', [])
        
        if not values:
            # print('No data found in the sheet.')
            return []
            
        # Filter for valid URLs and remove empty rows
        urls = [row[0] for row in values if row and row[0].startswith('http')]
        # print(f"Found {len(urls)} valid URLs in the sheet")
        return urls
        
    except Exception as e:
        print(f"Error fetching URLs from Google Sheets: {e}")
        return []

# Test the connection
resources = get_urls_from_sheet()

# ------ Part 2: Firecrawl Websites ------
# Initialize app
app = FirecrawlApp(api_key= os.environ.get("FIRECRAWL_KEY"))

# Crawl Website:
crawl_params = {
    'limit': 200,  # Limit to 200 pages
    'depth': 2,  # Crawl up to 2 levels deep from the start page
    'scrapeOptions': {
        'formats': ['html', 'markdown'],  # Save content in HTML and Markdown formats
        'customSelectors': {  # Specific elements to extract
            'titles': 'h1, h2',  # Extract main headings for structure
            'paragraphs': 'p',   # Extract paragraphs for main content
        }
    },
    'crawlDelay': 2,  # 2-second delay between requests
    'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',  # Mimic a common browser user agent
}

for link in resources:
    new_result = app.crawl_url(link)
    crawl_results.append(new_result)
    print(f"Crawled: {link}")  # Optional: print just the URL being crawled

# ------ Part 3: Write to JSON file  ------
# Save the crawl results to a JSON file
with open(JSON_FILE, 'w') as f:
    # We will re-write it each time.
    json.dump(crawl_results, f, indent=2)

print("Data has been written to json file.")
