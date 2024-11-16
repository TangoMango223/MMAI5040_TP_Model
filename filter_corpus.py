"""
filter_corpus.py
Goal: Filter out copyrighted domains from the existing corpus and save to a new JSON file
"""

import json
import logging
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Input and output file paths
INPUT_JSON = "torontopublicsafetycorpus.json"
OUTPUT_JSON = "non_copyrighted_torontopublicsafetycorpus.json"

def filter_corpus(input_file: str, output_file: str):
    """Filter out copyrighted domains and save to new JSON file."""
    
    # Define domains to exclude - using base domains without www
    excluded_domains = {
        'retailcouncil.org',
        'toronto.ca',
        'ttc.ca',
        # 'accsupport.com',
        # 'canadasmissing.ca',
        # 'mcsc.ca',
        # 'torontopolice.on.ca'  # Keep TPS domain
    }
    
    filtered_results = []
    excluded_count = 0
    kept_domains = set()
    
    try:
        # Read original JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # First, let's see all domains in the corpus
        all_domains = set()
        for batch in data:
            if 'data' in batch:
                for item in batch['data']:
                    if 'metadata' in item:
                        source_url = item['metadata'].get('sourceURL', 
                                   item['metadata'].get('url', ''))
                        try:
                            domain = urlparse(source_url).netloc.lower()
                            base_domain = '.'.join(domain.split('.')[-2:])
                            all_domains.add(base_domain)
                        except:
                            continue
        
        logger.info("\nAll domains in corpus:")
        for domain in sorted(all_domains):
            logger.info(f"- {domain}")
            
        # Now process and filter
        for batch in data:
            filtered_batch = batch.copy()
            filtered_items = []
            
            if 'data' in batch:
                for item in batch['data']:
                    if 'metadata' in item:
                        source_url = item['metadata'].get('sourceURL', 
                                   item['metadata'].get('url', ''))
                        
                        try:
                            domain = urlparse(source_url).netloc.lower()
                            base_domain = '.'.join(domain.split('.')[-2:])
                        except:
                            domain = ''
                            base_domain = ''
                            
                        if base_domain in excluded_domains:
                            excluded_count += 1
                            logger.info(f"Excluding document from: {domain} ({base_domain})")
                            continue
                            
                        filtered_items.append(item)
                        kept_domains.add(base_domain)
            
            if filtered_items:
                filtered_batch['data'] = filtered_items
                filtered_results.append(filtered_batch)
            
        # Save filtered results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2)
            
        logger.info(f"\nFiltering complete:")
        logger.info(f"- Excluded {excluded_count} documents")
        logger.info(f"\nKept domains:")
        for domain in sorted(kept_domains):
            logger.info(f"- {domain}")
        logger.info(f"\nSaved filtered corpus to {output_file}")
        
    except Exception as e:
        logger.error(f"Error filtering corpus: {str(e)}")

if __name__ == "__main__":
    filter_corpus(INPUT_JSON, OUTPUT_JSON) 