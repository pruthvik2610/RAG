from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from document_loader import load_documents

def create_vectorstore(file_path):
    docs = load_documents(file_path)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)
