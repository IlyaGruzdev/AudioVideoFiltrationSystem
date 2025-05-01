from fuzzywuzzy import fuzz

from tts import TTS
from stt import STT

commandsList = []
def equ(text, needed):
        return fuzz.ratio(text, needed) >= 70
def execute(text: str):
    print(f"> {text}")
        
    if equ(text, "расскажи анекдот"):
        text = "какой то анекдот!"
        tts.text2speech(text)
        print(f"- {text}")
        
    elif equ(text, "что ты умеешь"):
        text = "я умею всё, чему ты мен+я науч+ил!"
        tts.text2speech(text)
        print(f"- {text}")
        
    elif equ(text, "выключи"):
        text = "надеюсь, я не стану про+ектом, кот+орый ты забр+осишь!"
        tts.text2speech(text)
        print(f"- {text}")
        raise SystemExit
   
tts = TTS()
stt = STT(modelpath="vosk-model-ru-0.42")
print("listen...")
stt.listen(execute)