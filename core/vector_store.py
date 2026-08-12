import os 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document

CROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

def create_vector_store(transcription: str) -> Chroma:
    embedd_model = get_embedding_model()
    print("Creating vector store...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_text(transcription)

    docs = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

    vector_store = Chroma.from_documents(docs, embedd_model, persist_directory=CROMA_DIR, collection_name=COLLECTION_NAME)

    return vector_store

def load_vector_store() -> Chroma:  
    embedd_model = get_embedding_model()
    print("Loading vector store...")
    return Chroma(persist_directory=CROMA_DIR, embedding_function=embedd_model, collection_name=COLLECTION_NAME)

def get_retriever_from_vector_store(vector_store: Chroma, top_k: int = 5):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
