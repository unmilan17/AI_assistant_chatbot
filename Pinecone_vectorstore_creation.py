import arxiv

def search_arxiv(query, max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []

    # Iterate over results
    for result in search.results():
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'pdf_url': result.pdf_url,
            'published': result.published,
            'categories': result.categories
        }
        papers.append(paper_info)

    return papers

# Search for research papers related to "quantum computing"
papers = search_arxiv("machine learning", max_results=100)

count = 5

# Print the results
for idx, paper in enumerate(papers):
    if idx >= count:
        break
    print(f"Paper {idx + 1}: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Published: {paper['published']}")
    print(f"Categories: {', '.join(paper['categories'])}")
    print(f"Summary: {paper['summary']}")
    print(f"PDF URL: {paper['pdf_url']}")
    print("="*80)

import requests

# Function to download a paper's PDF from arXiv
def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {pdf_url}")

# Download the first paper's PDF
if papers:
  for idx, paper in enumerate(papers):
    pdf_url = papers[idx]['pdf_url']
    save_path = f"paper{idx + 1}.pdf"
    download_pdf(pdf_url, save_path)

def write_string_to_file(filename, content):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully wrote content to {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# print(text)  # Print first 500 characters

for i in range(1,101):
  pdf_path = f'paper{i}.pdf'
  text = extract_text_from_pdf(pdf_path)
  write_string_to_file(f'pdf_{i}.txt', text)

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



def split_text_with_recursive_splitter(text, chunk_size=512, chunk_overlap=50):
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be split.
        chunk_size (int): Maximum size of each chunk (in characters).
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Split text into chunks
chunks = split_text_with_recursive_splitter(text, chunk_size=50, chunk_overlap=10)

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader('/content', glob="**/*.txt")

docs = loader.load()

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize E5 embeddings
model_name = "intfloat/e5-large-v2"  # You can choose other E5 variants like "intfloat/e5-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from google.colab import userdata


pinecone_api_key = userdata.get("PINECONE_API_KEY")


# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Define your index name and embedding dimension (e.g., 1536 for many OpenAI models)
index_name = "llama-text-embed-v2-index"
dimension = 1024

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name='llama-text-embed-v2-index',
        dimension=1024,
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

from langchain_community.vectorstores import Pinecone
import pinecone
os.environ['PINECONE_API_KEY'] = userdata.get('PINECONE_API_KEY')
# index = pc.Index("llama-text-embed-v2-index")
vectorstore = PineconeVectorStore.from_documents(
  documents,
  index_name=index_name,
  embedding=embeddings,
)

batch_size = 50  # or smaller if you're still hitting limits

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )

# Get the index statistics
stats = index.describe_index_stats()

# Print the statistics
print(stats)

