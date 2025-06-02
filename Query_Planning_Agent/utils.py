from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

def create_vector_store(path):
    loader = TextLoader(path)
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embedding = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding)
