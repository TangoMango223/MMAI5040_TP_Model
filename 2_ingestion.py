"""
2_ingestion.py
Goal: Ingest scraped data into Pinecone Vector Database
"""

import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings 
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_json_data(file_path: str):
    """Load and parse the JSON file into LangChain documents."""
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        for batch in data:
            if 'data' in batch:
                for item in batch['data']:
                    if 'markdown' in item and 'metadata' in item:
                        # Get source URL from metadata
                        source_url = item['metadata'].get('sourceURL', 
                                   item['metadata'].get('url', ''))
                        
                        doc = Document(
                            page_content=item['markdown'],
                            metadata={
                                'source': source_url,  # Set URL as source for Pinecone visibility
                                'text': item['markdown'][:1000],  # Truncated preview is fine since we're chunking
                                'title': item['metadata'].get('title', '')
                            }
                        )
                        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents from JSON")
    return documents

def text_splitter(documents):
    """Split documents into smaller chunks."""
    total_splits = []    
    for doc in documents:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = splitter.split_documents([doc])
        total_splits.extend(splits)
    
    logger.info(f"Split into {len(total_splits)} chunks")
    return total_splits

def main():
    """Main function to process documents and load into Pinecone."""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Get index name from environment
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables")
    
    # Load documents from JSON
    documents = load_json_data('torontopublicsafetycorpus.json')
    
    # Split for storage later
    text_split_chunks = text_splitter(documents)
    
    if is_index_empty(os.environ["INDEX_NAME"]):
        logger.info("Ingesting documents into vector store...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        PineconeVectorStore.from_documents(text_split_chunks, embeddings, index_name=os.environ["INDEX_NAME"])
        logger.info("Document ingestion complete.")
    else:
        logger.info("Vector store already contains documents. Skipping ingestion.")

if __name__ == "__main__":
    main()

# Quick Math:
# Documents = 236 - uniquely scraped