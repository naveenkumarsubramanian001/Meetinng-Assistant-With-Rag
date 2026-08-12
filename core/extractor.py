from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

from core.summariser import get_model

def build_chain(system_prompt: str):
    model = get_model()

    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ])
        | model |StrOutputParser()
    )


def extract_action_items(transcribed_text: str) -> str:
    system_prompt = "You are a helpful assistant that extracts all action items from the meeting transcrip. you will provide taskname , task description , task owner and task deadline in a table format. If there are no action items, you will respond with 'No action items found.'"
    chain = build_chain(system_prompt)
    return chain.invoke(transcribed_text)


def extract_decisions(transcribed_text: str) -> str:
    system_prompt = "You are a helpful assistant that extracts all decisions made during the meeting. You will provide decision name, decision description, and decision owner in a table format. If there are no decisions, you will respond with 'No decisions found.'"
    chain = build_chain(system_prompt)
    return chain.invoke(transcribed_text)

def extract_questions(transcribed_text: str) -> str:
    system_prompt = "You are a helpful assistant that extracts all questions asked during the meeting. You will provide question name, question description, and question owner in a table format. If there are no questions, you will respond with 'No questions found.'"
    chain = build_chain(system_prompt)
    return chain.invoke(transcribed_text)


