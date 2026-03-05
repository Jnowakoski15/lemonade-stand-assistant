from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

loader = PyPDFLoader("Red_Hat_Tech_Menu_Watermarked.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
MENU_CHUNKS = [split.page_content for split in splits if len(split.page_content.strip()) > 10]
print(f"Chunks: {len(MENU_CHUNKS)}")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = embeddings_model.embed_documents(MENU_CHUNKS)
print(f"Embedding dimension: {len(embeddings[0])}")
