from av import  AudioFrame
import asyncio
from app.voice_recognition.voice_processed.main import AudioFrameProcessor
import numpy as np
from aiortc import AudioStreamTrack


class AudioTransformTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.echo_queue = asyncio.Queue()
        self.input_sample_rate = 48000  # Исходная частота
        self.frame_duration_ms = 20
        self.audio_processor = AudioFrameProcessor(track, self.echo_queue, self.input_sample_rate, self.frame_duration_ms)
      
      
        asyncio.ensure_future(self.audio_processor.processor_loop())

    async def recv(self):
        try:
            # Ждем эхо, иначе отправляем тишину
            chunk = await asyncio.wait_for(self.echo_queue.get(), timeout=1.0)
            frame = AudioFrame.from_ndarray(chunk, format="s16", layout="stereo")
        except asyncio.TimeoutError:
            empty = np.zeros((2, int(self.input_sample_rate * self.frame_duration_ms / 1000)), dtype=np.int16)
            frame = AudioFrame.from_ndarray(empty, format="s16", layout="stereo")

        frame.sample_rate = self.input_sample_rate
       
        return frame

   