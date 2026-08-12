from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

import os

load_dotenv()

_model = None

def get_model():
    global _model
    if _model is None:
        mistral_api_key = os.environ.get("MISTRALAI_API_KEY")
        if mistral_api_key is None:
            raise ValueError("MISTRALAI_API_KEY environment variable is not set.")
        _model = ChatMistralAI(model = "mistral-small-latest", api_key=mistral_api_key, temperature=0.7)
    return _model 


def split_transcribed_text(transcribed_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(transcribed_text)


def summarise_text(transcribed_text: str) -> str:
    model = get_model()

    map_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant that summarize this portion of text."),
        ("human", "{text}")]
    )

    map_chain = map_prompt | model | StrOutputParser()
    print("Splitting transcribed text into chunks...")
    chunks = split_transcribed_text(transcribed_text)
    
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    print("Combining chunk summaries into final summary...")


    combined_chunks ="\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that combine and summarize these portions of text into one final professional meeting summary in bullet points."),
        ("human", "{text}")
    ])

    print("Generating final summary...")
    combined_chain = combined_prompt | model | StrOutputParser()

    return combined_chain.invoke({"text": combined_chunks})

def generate_title( transripted_text: str) -> str:
    model = get_model()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) 
        | ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generate a short professional meeting title (maximum 8 words) for this meeting summary. return exactly the title without any additional text."),
            ("human", "{text}")
            ])
    | model | StrOutputParser()
    )
    print("Generating title for the meeting summary...")
    return title_chain.invoke({"text": transripted_text})

