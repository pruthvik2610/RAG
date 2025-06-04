from langchain.tools import Tool

def get_vectorstore_tool(vectorstore):
    return Tool(
        name="VectorDBRetriever",
        func=vectorstore.similarity_search,
        description="Useful to answer questions from documents"
    )
