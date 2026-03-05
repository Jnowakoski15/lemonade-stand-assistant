import os
import sys

def main():
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:19530")
    COLLECTION_NAME = "pizza_menu"
    
    try:
        from pymilvus import MilvusClient
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("Error: Missing required packages. Please run 'pip install -r requirements.txt'")
        sys.exit(1)

    # Read and chunk PDF
    try:
        print("Loading and splitting PDF...")
        loader = PyPDFLoader("Red_Hat_Tech_Menu_Watermarked.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(docs)
        
        # Extract purely the text from the document splits
        MENU_CHUNKS = [split.page_content.strip() for split in splits if len(split.page_content.strip()) > 10]
        print(f"Extracted {len(MENU_CHUNKS)} chunks from the PDF.")
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        sys.exit(1)

    print(f"Connecting to Milvus at {VECTOR_DB_URL}...")
    try:
        if VECTOR_DB_URL.startswith("http://") or VECTOR_DB_URL.startswith("https://"):
            client = MilvusClient(uri=VECTOR_DB_URL)
        else:
             # Use Milvus Lite (local file)
            print(f"Using Milvus Lite with local file: {VECTOR_DB_URL}")
            client = MilvusClient(VECTOR_DB_URL)
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        sys.exit(1)
    
    print("Loading HuggingFace embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Generating embeddings for menu items...")
    embeddings = embeddings_model.embed_documents(MENU_CHUNKS)
    dimension = len(embeddings[0])
    
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it for fresh upload...")
        client.drop_collection(collection_name=COLLECTION_NAME)
        
    print(f"Creating collection '{COLLECTION_NAME}' with dimension {dimension}...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=dimension,
        metric_type="COSINE",
        auto_id=True
    )
    
    data = []
    for i, text in enumerate(MENU_CHUNKS):
        data.append({
            "vector": embeddings[i],
            "text": text
        })
        
    print(f"Inserting {len(data)} items into '{COLLECTION_NAME}'...")
    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    
    # Flush makes data instantly searchable
    client.flush(collection_name=COLLECTION_NAME)
    print("Flushed collection.")
    
    print(f"Insertion successful! Inserted {res.get('insert_count', len(data))} rows.")
    print("Upload complete.")

if __name__ == "__main__":
    main()
