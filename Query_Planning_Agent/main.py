from retrievers.history_retriever import get_history_qa_chain
from retrievers.science_retriever import get_science_qa_chain
from planner import decompose_query
from aggregator import aggregate_answers

def get_subchain(sub_q):
    if "war" in sub_q.lower() or "empire" in sub_q.lower():
        return get_history_qa_chain()
    elif "physics" in sub_q.lower() or "energy" in sub_q.lower():
        return get_science_qa_chain()
    else:
        return get_science_qa_chain()  # fallback

def main():
    query = input("Enter your complex question: ")
    sub_questions = decompose_query(query)

    print(f"\n🧠 Decomposed into {len(sub_questions)} sub-questions:")
    for sq in sub_questions:
        print(" -", sq)

    answers = []
    for sq in sub_questions:
        chain = get_subchain(sq)
        answers.append(chain.run(sq))

    final_answer = aggregate_answers(answers)

    print("\n📝 Final Aggregated Answer:")
    print(final_answer)

if __name__ == "__main__":
    main()
