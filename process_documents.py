"""
process_documents.py
Goal: process documents, then put into Vector Store, "PineCone"
"""

# Import Statements:
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import PyPDF2

# Load environment variables
load_dotenv()

# Set up logging - only show INFO and above, remove timestamp for cleaner output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# -------------- Part 1: Functions and Classes -------------- #

class EncodingMarkdownLoader:
    """
    Class created to handle loading for markdown files, originally web pages.
    """
    def __init__(self, file_path):
        """
        Some doc string here
        """
        self.file_path = file_path

    def load(self):
        """
        Some doc string here
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

def load_pdf(file_path):
    """
    Handle loading using langchain loaders, for PDF files.
    """
    loaders = [PDFMinerLoader, PyPDFLoader, UnstructuredPDFLoader]
    for loader_class in loaders:
        try:
            loader = loader_class(file_path)
            return loader.load()
        except Exception:
            pass
    logger.error(f"All PDF loaders failed for {file_path}")
    return []

def load_documents(directory):
    """
    Handle document loading for both PDF and Markdown files, from a directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, directory)
    
    if not os.path.exists(downloads_dir):
        logger.error(f"The directory {downloads_dir} does not exist.")
        return []
    
    documents = []
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    for filename in os.listdir(downloads_dir):
        total_files += 1
        file_path = os.path.join(downloads_dir, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1].lower()
            try:
                if file_extension == '.pdf':
                    file_documents = load_pdf(file_path)
                    processed_files += 1
                elif file_extension == '.md':
                    loader = EncodingMarkdownLoader(file_path)
                    file_documents = loader.load()
                    processed_files += 1
                else:
                    logger.info(f"Skipping unsupported file: {filename}")
                    skipped_files += 1
                    continue
                
                # Update metadata with correct source URL
                for doc in file_documents:
                    if file_extension == '.pdf':
                        # For PDFs, read the SourceURL from metadata
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            source_url = pdf_reader.metadata.get('/SourceURL', file_path)
                    elif file_extension == '.md':
                        # For Markdown, read the first line for the source URL
                        with open(file_path, 'r', encoding='utf-8') as md_file:
                            first_line = md_file.readline().strip()
                            source_url = first_line.replace('<!-- Source URL: ', '').replace(' -->', '') if first_line.startswith('<!-- Source URL:') else file_path
                    
                    doc.metadata['source'] = source_url
                
                documents.extend(file_documents)
                logger.info(f"Successfully processed: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                skipped_files += 1
    
    logger.info(f"Total files in directory: {total_files}")
    logger.info(f"Successfully processed files: {processed_files}")
    logger.info(f"Skipped or errored files: {skipped_files}")
    logger.info(f"Total documents loaded: {len(documents)}")
    
    return documents


def text_splitter(documents):
    total_splits = []    
    for doc in documents:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,  # Reduced from 2000
            chunk_overlap=200,  # Reduced from 300
            length_function=len,
        )
        
        splits = splitter.split_documents([doc])
        total_splits.extend(splits)
    return total_splits


def is_index_empty(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    return stats.total_vector_count == 0


def process_documents():
    """
    Handle document processing, then put into Vector Store, "PineCone"
    """
    documents = load_documents("downloads")
    logger.info(f"Processed {len(documents)} documents")
    
    text_split_chunks = text_splitter(documents)
    
    if is_index_empty(os.environ["INDEX_NAME"]):
        logger.info("Ingesting documents into vector store...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        PineconeVectorStore.from_documents(text_split_chunks, embeddings, index_name=os.environ["INDEX_NAME"])
        logger.info("Document ingestion complete.")
    else:
        logger.info("Vector store already contains documents. Skipping ingestion.")

# -------------- Part 2: Main Control -------------- #

# If running this file directly...
if __name__ == "__main__":
    process_documents()
