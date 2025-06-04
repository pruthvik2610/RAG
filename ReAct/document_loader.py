from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def load_documents(file_path):
    loader = TextLoader(file_path)
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(raw_docs)
