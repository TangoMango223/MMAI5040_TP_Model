""" 
scrape_torontopolice.py
Goal: Scraping from Crime Prevention Articles (and other sources), using FireCrawl.

# Scrape non-copyrighted sources from the Google Sheet, fixing corpus for the OpenData Competition.
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
from datetime import datetime
import pytz

# FireCrawl App:
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv(".env", override=True)

# Variables:
JSON_FILE = "non_copyrighted_torontopublicsafetycorpus.json"
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

#check_and_install_requirements()


# ------ Connect to Google Sheets again ------

# Configuration
SPREADSHEET_ID = '1PEVuqlUrvVoJJUflYGq4gN6itj9WZzGB-UHYnsIusng'
RANGE_NAME = 'Sheet2!C:C'
SERVICE_ACCOUNT_FILE = 'torontopolicellmbot-9644404c0bdf.json'

def initialize_sheets_service():
    """Initialize and return the Google Sheets service"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        service = build('sheets', 'v4', credentials=credentials)
        return service  # Return the entire service object
    except Exception as e:
        print(f"Error initializing Google Sheets service: {e}")
        return None

def get_urls_from_sheet():
    """Fetch URLs from Google Sheet with proper error handling"""
    try:
        service = initialize_sheets_service()
        if not service:
            return []
            
        # Use the full service path
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            print('No data found in the sheet.')
            return []
            
        # Filter for valid URLs and remove empty rows
        urls = [row[0] for row in values if row and row[0].startswith('http')]
        print(f"Found {len(urls)} valid URLs in the sheet")
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
    'maxPages': 100,  # Reduced from original 200
    'maxDepth': 2,   # Reduced from original 2
    'scrapeOptions': {
        'format': ['markdown'],
        'selectors': {
            'titles': 'h1, h2',
            'paragraphs': 'p'
        }
    },
    'delay': 5,      # Changed from 'crawlDelay' to 'delay'
    'agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'  # Changed from 'userAgent' to 'agent'
}

def extract_wait_time(error_message):
    """Extract wait time from FireCrawl error message"""
    wait_time_match = re.search(r'retry after (\d+)s', str(error_message))
    if wait_time_match:
        return int(wait_time_match.group(1))
    return 60

def crawl_with_retry(app, link, max_retries=3):
    """
    Attempt to crawl a URL with retry logic for rate limiting
    """
    for attempt in range(max_retries):
        try:
            result = app.crawl_url(link, params=crawl_params)
            print(f"Successfully crawled: {link}")
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Payment Required - Stop immediately
            if "Payment Required" in error_str:
                print(f"Insufficient credits for {link}. Stopping all crawls...")
                return "STOP_ALL"
                
            # Rate limit - Wait the full duration before next attempt
            elif "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = extract_wait_time(error_str)
                    print(f"Rate limit hit. Waiting full duration of {wait_time} seconds")
                    print(f"Error details: {error_str}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to crawl {link} after {max_retries} attempts")
                    return None
            else:
                print(f"Error crawling {link}: {error_str}")
                return None
    
    return None

# Main crawling loop with proper wait handling
print(f"Starting to crawl {len(resources)} URLs...")
for link in resources:
    print(f"\nStarting crawl for: {link}")
    result = crawl_with_retry(app, link)
    
    # Check for stop condition
    if result == "STOP_ALL":
        print("Stopping all crawls due to insufficient credits")
        break
        
    if result is not None:
        crawl_results.append(result)
        print(f"Successfully added results for: {link}")
        time.sleep(5)  # Small delay between successful crawls
    else:
        print(f"Skipping {link} due to errors")
        time.sleep(2)  # Minimal delay after errors

# Save results only if we have any
if crawl_results:
    with open(JSON_FILE, 'w') as f:
        json.dump(crawl_results, f, indent=2)
    print(f"\nData has been written to {JSON_FILE}")
    print(f"Successfully crawled {len(crawl_results)} URLs")
else:
    print("\nNo results were collected during crawling")
