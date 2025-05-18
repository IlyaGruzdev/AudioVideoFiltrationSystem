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
from scipy.signal import resample
# class STT:
#     def __init__(self, modelpath: str = "model", sample_rate: int = 16000):
#         self.model = vosk.Model(modelpath)
#         self.recognizer = None # Инициализируем при первом вызове process_frame
#         self.sample_rate = sample_rate

#     async def process_frame(self, frame: AudioFrame):
    

#         if self.recognizer is None:
#             self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
#             audio_data = frame.to_ndarray()
#             print(f" Sample rate: {frame.sample_rate}, self:{self.sample_rate}")    
#             if audio_data.dtype != np.int16:
#                 audio_data = np.int16(audio_data * 32767)
#             audio_data_resampled = resample(audio_data, int(len(audio_data) * 16000 / 48000))

#             with wave.open('first_frame.wav', 'wb') as wf:
#                 wf.setnchannels(2) # Устанавливаем количество каналов
#                 wf.setsampwidth(2) # 2 байта на сэмпл (16 бит)
#                 wf.setframerate(frame.sample_rate)
#                 wf.writeframesraw(audio_data_resampled.tobytes())
#         # Преобразование фрейма в формат, понятный Vosk
       
#         data = self._convert_frame_to_vosk_format(frame)
#         if self.recognizer.AcceptWaveform(data):
#             result = json.loads(self.recognizer.Result())
#             # if "text" in result and result["text"]: # Проверка на пустую строку
#             return result["text"]
#         elif self.recognizer.PartialResult(): # Промежуточные результаты
#           result = json.loads(self.recognizer.PartialResult())
#           if "partial" in result and result["partial"]: # Проверка на пустую строку
#                 return result["partial"]

#         return None # Возвращаем None, если распознавания не произошло
    
#     def _convert_frame_to_vosk_format(self, frame: AudioFrame):
#         """Преобразует AudioFrame (из библиотеки av) в формат, подходящий для Vosk."""

#         # 1. Получаем данные в виде numpy array
#         audio_data = frame.to_ndarray()

#         # 2. Преобразуем в int16 с масштабированием, если нужно
#         if audio_data.dtype != np.int16:
#             audio_data = np.int16(audio_data * 32767)
#         buf = []
#         for i, item in audio_data.reshape(-1):
#             if(i%2 == 0):
#                 buf.append(item)
#         audio_data = buf
#         # audio_data_resampled = resample(audio_data, int(len(audio_data) * 16000 / 48000))
        
#         # 3. Преобразуем в байты
#         audio_bytes = audio_data.tobytes()
#         return audio_bytes
class STT:
    def __init__(self, modelpath: str = "model", sample_rate: int = 8000): # Убедитесь, что это 8000
        self.model = vosk.Model(modelpath)
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        print(f"STT_LOG: Initialized with sample rate: {self.sample_rate}")

    async def process_frame(self, frame: AudioFrame):
        print(f"STT_LOG: process_frame called. Frame SR: {frame.sample_rate}, Layout: {frame.layout}, Samples: {frame.samples}")
        if frame is None:
            print("STT_LOG: Received None frame, skipping.")
            return None

        # Проверяем, соответствует ли частота дискретизации фрейма той, что ожидает распознаватель
        if frame.sample_rate != self.sample_rate:
            print(f"STT_LOG: ERROR - Frame sample rate ({frame.sample_rate}) "
                  f"does not match recognizer sample rate ({self.sample_rate}). "
                  "Resampling should have happened before this point.")
            # Здесь можно либо возбудить исключение, либо попытаться проигнорировать/обработать,
            # но это указывает на проблему в вызывающем коде.
            return "Error: Sample rate mismatch"

        audio_data_for_vosk = frame.to_ndarray().tobytes() # Должен быть уже моно, s16, нужная SR

        # Отладочная запись в WAV
        try:
            with wave.open('debug_audio_for_vosk.wav', 'wb') as wf:
                wf.setnchannels(1) # Моно
                wf.setsampwidth(frame.format.bytes) # 
                wf.setframerate(frame.sample_rate) # Должно быть self.sample_rate
                wf.writeframesraw(audio_data_for_vosk)
            print("STT_LOG: Saved debug_audio_for_vosk.wav")
        except Exception as e:
            print(f"STT_LOG: Error saving debug WAV: {e}")

        recognized_text = None
        if self.recognizer.AcceptWaveform(audio_data_for_vosk):
            result_json = self.recognizer.Result()
            print(f"STT_LOG: Vosk Result: {result_json}")
            result = json.loads(result_json)
            if "text" in result and result["text"]:
                recognized_text = result["text"]
        else:
            partial_result_json = self.recognizer.PartialResult()
            print(f"STT_LOG: Vosk PartialResult: {partial_result_json}")
            partial_result = json.loads(partial_result_json) # Можно обработать и частичные
            if "partial" in partial_result and partial_result["partial"]:
                return partial_result

        return recognized_text