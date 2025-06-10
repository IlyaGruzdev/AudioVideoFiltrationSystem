from fuzzywuzzy import fuzz
from app.voice_recognition.voice_processed.stt import STT
from app.voice_recognition.voice_processed.audio_filter import AudioFilter
import resampy
import asyncio
import os
import numpy as np
from av import  AudioFrame
from fractions import Fraction

cur_path = os.path.dirname(os.path.abspath(__file__))
class AudioFrameProcessor:
    def __init__(self, track, echo_queue, input_sample_rate, frame_duration_ms):
       self.track = track
       self.echo_queue = echo_queue
       self.input_sample_rate = input_sample_rate
       self.frame_duration_ms = frame_duration_ms
       self.stt = STT(modelpath=cur_path+"/vosk-model-ru-0.42", sample_rate=16000)
       self.processing = False
       self.started = False
       self.buffer = []
       self.auido_filter = AudioFilter()
       
    async def processor_loop(self):
        print("Audio processor loop started")
        while True:
            frame = await self.track.recv()
            data = frame.to_ndarray()

            # Stereo → mono
            if frame.layout.name == "stereo":
                if data.ndim == 2 and data.shape[0] == 1:
                    interleaved = data.flatten()
                    reshaped = interleaved.reshape(-1, 2)
                    mono = reshaped.mean(axis=1).astype(np.int16)
                elif data.ndim == 2 and data.shape[0] == 2:
                    mono = data.mean(axis=0).astype(np.int16)
                else:
                    continue
            elif frame.layout.name == "mono":
                mono = data.flatten().astype(np.int16)
            else:
                print("Unsupported layout:", frame.layout)
                continue

            # Ресемплируем в 16000
            mono_16k = resampy.resample(mono.astype(np.float32), frame.sample_rate, self.stt.sample_rate).astype(np.int16)
            self.buffer.append(mono_16k)

            total = sum(len(b) for b in self.buffer)
            if total >= self.stt.sample_rate * 5 and not self.processing:#5s
                self.processing = True
                asyncio.create_task(self.process_and_echo(np.concatenate(self.buffer)))
                self.buffer = []

    async def process_and_echo(self, audio_data):
        print("STT: Enough audio collected. Sending to recognizer...")
        frame = AudioFrame.from_ndarray(audio_data.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.stt.sample_rate
        frame.time_base = Fraction(1, self.stt.sample_rate)
        frame.pts = 0

        result = await self.stt.recognize_text(frame)
        print("STT result:", result)

        # Вернем стерео с SR 48000
        upsampled = resampy.resample(audio_data.astype(np.float32), self.stt.sample_rate, self.input_sample_rate).astype(np.int16)
        stereo = np.stack([upsampled, upsampled], axis=0)

        samples_per_chunk = int(self.input_sample_rate * self.frame_duration_ms / 1000)
        for i in range(0, stereo.shape[1], samples_per_chunk):
            chunk = stereo[:, i:i + samples_per_chunk]
            if chunk.shape[1] < samples_per_chunk:
                pad = np.zeros((2, samples_per_chunk - chunk.shape[1]), dtype=np.int16)
                chunk = np.hstack([chunk, pad])
            await self.echo_queue.put(chunk)

        self.processing = False
