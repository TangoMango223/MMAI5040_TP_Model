"""
2_ingestion.py
Goal: Ingest non-copyrighted data into Pinecone Vector Database 'torontopolice2'
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
load_dotenv(dotenv_path=".env", override=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Hard-code the index name
PINECONE_INDEX = "torontopolice2"
INPUT_FILE = "non_copyrighted_torontopublicsafetycorpus.json"

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
                                'source': source_url,
                                'text': item['markdown'][:1000],
                                'title': item['metadata'].get('title', '')
                            }
                        )
                        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents from {file_path}")
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
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Load documents from non-copyrighted JSON
    logger.info(f"Loading documents from {INPUT_FILE}...")
    documents = load_json_data(INPUT_FILE)
    
    # Split text into chunks
    text_split_chunks = text_splitter(documents)
    
    logger.info(f"Ingesting documents into vector store index '{PINECONE_INDEX}'...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create vector store with hardcoded index name
    vector_store = PineconeVectorStore.from_documents(
        documents=text_split_chunks, 
        embedding=embeddings, 
        index_name=PINECONE_INDEX
    )
    logger.info(f"Document ingestion complete into index: {PINECONE_INDEX}")

if __name__ == "__main__":
    main()

# Quick Math:
# Documents = 236 - uniquely scraped