from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import create_vector_store

def get_history_qa_chain():
    vectordb = create_vector_store("docs/history_docs.txt")
    llm = ChatOpenAI()
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
