from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.translator import translate_all_chunks
from utils.check_language import find_language
from core.extractor import extract_action_items, extract_decisions, extract_questions


source= "https://youtu.be/wAr_t2OsEdc"

chunks = process_input(source)
transcription = transcribe_all(chunks, translate=False)

language = find_language(transcription)
if language == "unknown":
    print("The language of the transcription is unknown or not supported.")
elif language == "ta":
    print("The transcription is in Tamil. Translating to English...")
    transcription = translate_all_chunks([transcription])
else:
    print("The transcription is in English. No translation needed.")    

print(transcription)

actions_items = extract_action_items(transcription)
print("Action Items:")
print(actions_items)

decisions = extract_decisions(transcription)
print("Decisions:")
print(decisions)    

questions = extract_questions(transcription)      
print("Questions:")
print(questions)

