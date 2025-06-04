from react_agent import run_react_agent

if __name__ == "__main__":
    query = input("Ask your question: ")
    response = run_react_agent(query)
    print("\n🔍 Response:\n", response)
