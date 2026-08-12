import os 

from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import create_vector_store, load_vector_store, get_retriever_from_vector_store

_model = None

def get_model():
    global _model
    if _model is None:
        mistral_api_key = os.environ.get("MISTRALAI_API_KEY")
        if mistral_api_key is None:
            raise ValueError("MISTRALAI_API_KEY environment variable is not set.")
        _model = ChatMistralAI(model = "mistral-small-latest", api_key=mistral_api_key, temperature=0.7)
    return _model 


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(system_prompt: str):
    model = get_model()

    vector_store = load_vector_store()
    retriever = get_retriever_from_vector_store(vector_store)
    question_input = RunnableLambda(lambda inputs: inputs["question"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a expert assistant that answers questions only based on the provided context.If answer is not found in the context, respond with 'I don't know'.alwasy be concise and professional. context from transcription is {context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": question_input | retriever | RunnableLambda(format_docs), "question": question_input}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Asking question: {question}")
    answer = rag_chain.invoke({"question": question})
    print(f"Answer: {answer}") 
    return answer 