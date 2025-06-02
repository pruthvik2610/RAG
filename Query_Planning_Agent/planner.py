from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def decompose_query(query):
    prompt = PromptTemplate.from_template(
        """Break down the following complex question into simpler sub-questions:
Question: {query}

Sub-questions:"""
    )
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query)
    return [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
