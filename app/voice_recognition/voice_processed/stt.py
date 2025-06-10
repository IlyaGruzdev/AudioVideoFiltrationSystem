import vosk
import io
import json
import wave
import numpy as np
import sys
import numpy as np # Для преобразования данных
from av import AudioFrame
from scipy.signal import resample

class STT:
    def __init__(self, modelpath: str = "model", sample_rate: int = 8000): # Убедитесь, что это 8000
        self.model = vosk.Model(modelpath)
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        print(f"STT_LOG: Initialized with sample rate: {self.sample_rate}")

    async def recognize_text(self, frame: AudioFrame):
        audio_data_for_vosk = self.frame_to_bytes(frame)
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
    
    def frame_to_bytes(self, frame):
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
        return audio_data_for_vosk
