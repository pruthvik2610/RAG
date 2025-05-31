from retrievers.code_retriever import get_code_qa_chain
from retrievers.legal_retriever import get_legal_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from router import route_query

def main():
    query = input("Ask your question: ")
    route = route_query(query)

    if route == "legal":
        qa_chain = get_legal_qa_chain()
    elif route == "code":
        qa_chain = get_code_qa_chain()
    else:
        # Default LLM with no RAG
        prompt = PromptTemplate.from_template("Answer the following question:\n{question}")
        llm = ChatOpenAI()
        qa_chain = LLMChain(llm=llm, prompt=prompt)

    result = qa_chain.run(query)
    print(f"\n🔍 Routed to: {route.upper()} retriever\n📝 Answer: {result}")

if __name__ == "__main__":
    main()
