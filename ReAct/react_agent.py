from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from create_vectorstore import create_vectorstore
from retriever_tool import get_vectorstore_tool
import config

def run_react_agent(query: str):
    # Load Vector DB
    vectorstore = create_vectorstore("data.txt")
    
    # Tool
    retriever_tool = get_vectorstore_tool(vectorstore)

    # LLM
    llm = ChatOpenAI(temperature=0, model_name=config.MODEL_NAME)

    # Agent
    agent = initialize_agent(
        tools=[retriever_tool],
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )
    return agent.run(query)
