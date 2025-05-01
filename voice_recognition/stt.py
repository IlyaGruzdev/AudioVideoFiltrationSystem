# import sounddevice as sd
# import vosk
# import sys
# import queue
# import json

# class STT:
#     def __init__(self, modelpath: str = "model", samplerate: int = 16000):
#         self.__REC__ = vosk.KaldiRecognizer(vosk.Model(modelpath), samplerate)
#         self.__Q__ = queue.Queue()
#         self.__SAMPLERATE__ = samplerate

    
#     def q_callback(self, indata, _, __, status):
#         if status:
#             print(status, file=sys.stderr)
#         self.__Q__.put(bytes(indata))

#     def listen(self, executor: callable):
#         with sd.RawInputStream(
#                 samplerate=self.__SAMPLERATE__, 
#                 blocksize=8000, 
#                 device=5, 
#                 dtype='int16',
#                 channels=1, 
#                 callback=self.q_callback
#             ):
#             while True:
#                 data = self.__Q__.get()
#                 if self.__REC__.AcceptWaveform(data):
#                     result = json.loads(self.__REC__.Result())["text"]
#                     executor(result)
import vosk
import io
import json
import wave
import numpy as np
import sys
import numpy as np # Для преобразования данных
from av import AudioFrame
class STT:
    def __init__(self, modelpath: str = "model", sample_rate: int = 16000):
        self.model = vosk.Model(modelpath)
        self.recognizer = None # Инициализируем при первом вызове process_frame
        self.sample_rate = sample_rate

    async def process_frame(self, frame: AudioFrame):
        """
        Распознает речь в переданном аудиофрейме.

        Args:
            frame: Аудиофрейм (aiortc.AudioFrame).
        """

        if self.recognizer is None:
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

        # Преобразование фрейма в формат, понятный Vosk
       
        data = frame.to_ndarray().tobytes()

        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            # if "text" in result and result["text"]: # Проверка на пустую строку
            return result["text"]
        elif self.recognizer.PartialResult(): # Промежуточные результаты
          result = json.loads(self.recognizer.PartialResult())
          if "partial" in result and result["partial"]: # Проверка на пустую строку
                return result["partial"]

        return None # Возвращаем None, если распознавания не произошло
    
    def _convert_frame_to_wav(self, frame: AudioFrame):
        """Преобразует AudioFrame в байты WAV."""
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(frame.channels)
                wf.setsampwidth(2) # 16-bit PCM
                wf.setframerate(frame.sample_rate)
                # Ключевое изменение: преобразование в int16 с правильным масштабированием
                audio_int16 = (np.int16(frame.to_ndarray() * 32767)).tobytes() # Масштабирование
                wf.writeframesraw(audio_int16)
            return wav_buffer.getvalue()