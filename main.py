from core.transcriber import transcribe_all
from core.translator import translate_all_chunks
from core.extractor import extract_action_items, extract_decisions, extract_questions
from utils.audio_processor import process_input
from utils.check_language import find_language
from core.summariser import summarise_text, generate_title
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str) -> dict:
    chunks = process_input(source)
    transcription = transcribe_all(chunks, translate=False)

    language = find_language(transcription)
    if language == "unknown":
        print("The language of the transcription is unknown or not supported.")
        return
    elif language == "ta":
        print("The transcription is in Tamil. Translating to English...")
        transcription = translate_all_chunks([transcription])
    else:
        print("The transcription is in English. No translation needed.")    

    actions_items = extract_action_items(transcription)
    decisions = extract_decisions(transcription)
    questions = extract_questions(transcription)      
    summary = summarise_text(transcription)
    title = generate_title(transcription)
    rag_chain = build_rag_chain(transcription)

    return {
        "title": title,
        "summary": summary,
        "transcription": transcription,
        "action_items": actions_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    source = "https://youtu.be/3WrZMzqpFTc"
    results = run_pipeline(source)
    
    print("Title:")
    print(results["title"])
    print("\nSummary:")
    print(results["summary"])
    print("\nTranscription:")

    print(results["transcription"])
    print("\nAction Items:")
    print(results["action_items"])
    print("\nDecisions:")
    print(results["decisions"])
    print("\nQuestions:")       
    print(results["questions"])

    print("\nChat with RAG Engine:")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break
        if not user_input:
            print("Please enter a valid question or type 'exit' to quit.")
            continue
        response = ask_question(results["rag_chain"], user_input)
        print(f"RAG Engine: {response}")