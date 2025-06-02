from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def aggregate_answers(answers):
    joined = "\n".join([f"Sub-Answer {i+1}: {a}" for i, a in enumerate(answers)])
    prompt = PromptTemplate.from_template(
        """Combine the following sub-answers into a final comprehensive answer:

{sub_answers}

Final Answer:"""
    )
    llm = ChatOpenAI()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"sub_answers": joined})
