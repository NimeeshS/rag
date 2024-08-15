import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from tqdm import tqdm

CHROMA_PATH = "chroma/"
DATA_PATH = "data/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resetchroma", action="store_true", help="Reset the chroma folder.")
    parser.add_argument("--resetdata", action="store_true", help="Reset the data folder.")
    args = parser.parse_args()
    run = True
    if args.resetchroma:
        print("✨ Clearing Chroma")
        clear_chroma()
        run = False
    if args.resetdata:
        print("✨ Clearing Data")
        clear_data()
        run = False

    # Create (or update) the data store.
    if run:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

def load_documents():
    documents = []
    
    # Load PDF files
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents.extend(pdf_loader.load())
    
    # Load TXT files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            txt_loader = TextLoader(os.path.join(DATA_PATH, filename))
            documents.extend(txt_loader.load())
    
    # Load MD files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):
            md_loader = UnstructuredMarkdownLoader(os.path.join(DATA_PATH, filename))
            documents.extend(md_loader.load())

    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        # Use tqdm to display progress bar
        for chunk in tqdm(new_chunks, desc="👉 Adding new documents", unit="chunk"):
            db.add_documents([chunk], ids=[chunk.metadata["id"]])

    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_chroma():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        os.mkdir(CHROMA_PATH)

def clear_data():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        os.mkdir(DATA_PATH)

if __name__ == "__main__":
    main()
