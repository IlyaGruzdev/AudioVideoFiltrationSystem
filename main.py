from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from src.schemas import Offer
import argparse
import asyncio
import numpy as np
import json
import logging
import os
import platform
import ssl
import cv2
from av import VideoFrame, AudioFrame
from fractions import Fraction
import wave
import io
from scipy.signal import resample
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaBlackhole
from aiortc.rtcrtpsender import RTCRtpSender
from voice_recognition.stt import STT

ROOT = os.path.dirname(__file__)

app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name="static")
templates = Jinja2Templates(directory="templates")
relay = None
webcam = None

stt = STT(modelpath="voice_recognition/vosk-model-ru-0.42", sample_rate=16000)

faces = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
smiles = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")


class AudioTransfromTrack(AudioStreamTrack):
    kind = "audio"
    def __init__(self, track, stt: STT):
        super().__init__()
        self.track = track
        self.stt = stt
        self.audio_buffer = [] # Буфер для накопления аудиоданных
        self.buffer_duration = 0 # Текущая длительность буфера в секундах
        self.target_duration = 5 # Целевая длительность буфера (5 секунд)
        self._sample_rate = None # Для хранения частоты дискретизации

    
        # combined_audio = []
        # combined_samples = 0
        # target_duration = 5 # Целевая длительность в секундах
        # sample_rate = None
        # first_frame_format = None
        # first_frame_layout = None


        # while True:
        #     frame = await self.track.recv()

        #     if sample_rate is None:
        #         frame.time_base = Fraction(1, frame.sample_rate)
        #         sample_rate = frame.sample_rate
        #         first_frame_format = frame.format
        #         first_frame_layout = frame.layout
        #         print("sampale_rate:", sample_rate, first_frame_format, first_frame_layout)
        #         print(frame.to_ndarray().reshape(-1)[:30])

        #     audio_data = frame.to_ndarray()
        #     if audio_data.ndim > 1:
        #         audio_data = audio_data.reshape(-1)
        #     if audio_data.dtype != np.int16:
        #         audio_data = np.int16(audio_data * 32767)
        #     combined_audio.append(audio_data)
        #     combined_samples += frame.samples


        #     if (combined_samples / sample_rate) >= target_duration:
        #         break

        # combined_audio = np.concatenate(combined_audio)

        # # Создание нового AudioFrame
        # combined_frame = AudioFrame(
        #     format=first_frame_format, layout=first_frame_layout, samples=combined_samples
        # )
        
        # print(combined_audio[:30])
        # combined_audio_bytes = combined_audio.tobytes()
        # for plane in combined_frame.planes:
        #     required_bytes = plane.buffer_size 
        #     plane.update(combined_audio_bytes[:required_bytes]) # Записываем данные
        #     combined_audio_bytes = combined_audio_bytes[required_bytes:] # Удаляем 
        # combined_frame.time_base = Fraction(1, sample_rate)
        # combined_frame.pts = int(combined_samples * Fraction(1, sample_rate) / combined_frame.time_base)
        # combined_frame.sample_rate = sample_rate
        # text = await self.stt.process_frame(combined_frame)
        # print("text:", text)
        # # Дальнейшая обработка combined_frame
        # # ... (ваш код, например, распознавание или сохранение в WAV)

        # return combined_frame # Возвращаем
    async def set_track(self, track): # Если трек устанавливается позже
        self.track = track
        print("recv_processor: Audio track has been set.")
        
    async def recv(self):
        print("recv_processor: Starting recv method...")
        if self.track is None:
            print("recv_processor: ERROR - self.track is None. Cannot proceed.")
            return None # Или возбудить исключение

        combined_audio_processed = []
        total_processed_samples = 0
        target_duration_seconds = 5

        target_sample_rate_vosk = self.stt.sample_rate
        target_layout_vosk = "mono"
        target_format_vosk = "s16"

        first_frame_received = False
        input_sample_rate = None
        input_original_layout_str = None # Для отладки
        iteration_count = 0
        all_processed_mono_chunks_input_sr = []
        all_original_mono_chunks_for_debug = []
        nframe = []
        print(f"STT configured for sample rate: {target_sample_rate_vosk}")

        while True:
            iteration_count += 1
            print(f"recv_processor: Iteration {iteration_count}, attempting to receive frame...")
            try:
                frame = await self.track.recv()
                if frame is None:
                    print("recv_processor: Received None frame (end of track?). Breaking loop.")
                    break
                print(f"recv_processor: Frame {iteration_count} received successfully.")
            except Exception as e:
                print(f"recv_processor: Error receiving frame in iteration {iteration_count}: {e}")
                await asyncio.sleep(0.05)
                continue

            if not first_frame_received:
                input_sample_rate = frame.sample_rate
                input_original_layout_str = str(frame.layout) # Сохраняем как строку для сравнения
                print(f"Input audio (first frame): SR={input_sample_rate}, Format={frame.format}, Layout={input_original_layout_str}")
                first_frame_received = True


            audio_data_original = frame.to_ndarray()
            current_frame_layout_str = str(frame.layout)
            print(f"recv_processor: Frame {iteration_count} - Original data shape: {audio_data_original.shape}, dtype: {audio_data_original.dtype}, layout: {current_frame_layout_str}")
            if audio_data_original.dtype != np.int16:
                print(f"recv_processor: Frame {iteration_count} - Converting dtype from {audio_data_original.dtype} to int16.")
                if np.issubdtype(audio_data_original.dtype, np.floating):
                    audio_data_original = (audio_data_original * 32767).astype(np.int16)
                else: # Если это другой целый тип, может потребоваться иное преобразование
                    print(f"recv_processor: Frame {iteration_count} - Warning: Unexpected dtype {audio_data_original.dtype} for non-float conversion to int16.")
                    audio_data_original = audio_data_original.astype(np.int16) # Попытка прямого преобразования
            
            # Обработка аудио данных в зависимости от их структуры и layout
            nframe.append(audio_data_original.copy())
            processed_mono_audio_chunk = None
            if current_frame_layout_str == "<av.AudioLayout 'stereo'>":

                if audio_data_original.ndim == 2 and audio_data_original.shape[0] == 1:
                    # Случай (1, N) для стерео: L1,R1,L2,R2... в audio_data_original[0]
                    interleaved_stereo_data = audio_data_original.flatten() # Становится 1D: [L1,R1,L2,R2,...]
                    if len(interleaved_stereo_data) % 2 != 0:
                        print(f"recv_processor: Frame {iteration_count} - Warning! Odd number of samples ({len(interleaved_stereo_data)}) after flattening stereo (1, N) data. Skipping frame.")
                        continue
                    # Преобразуем в (samples_per_channel, 2)
                    num_samples_per_channel = len(interleaved_stereo_data) // 2

                    if num_samples_per_channel > 0:
                        if num_samples_per_channel * 2 != len(interleaved_stereo_data):
                             print(f"STEREO_DEBUG: Frame {iteration_count} - ERROR! Reshape dimension mismatch. num_samples_per_channel*2 ({num_samples_per_channel*2}) != len_interleaved ({len(interleaved_stereo_data)})")
                             continue # Пропустить этот фрейм

                        reshaped_stereo = interleaved_stereo_data.reshape(num_samples_per_channel, 2)
                        processed_mono_audio_chunk = reshaped_stereo.mean(axis=1).astype(np.int16)
                        print(f"recv_processor: Frame {iteration_count} - Stereo (1,N) to mono. Chunk shape: {processed_mono_audio_chunk.shape}")
                    else:
                        processed_mono_audio_chunk = np.array([], dtype=np.int16)

                elif audio_data_original.ndim == 2 and audio_data_original.shape[1] == 2:
                    # Стандартный случай стерео (samples, 2)
                    print(f"STEREO_DEBUG: Frame {iteration_count} - Input (N,2) stereo. frame.samples: {frame.samples}, audio_data_original.shape[0]: {audio_data_original.shape[0]}")
                    if frame.samples != audio_data_original.shape[0]:
                        print(f"STEREO_DEBUG: Frame {iteration_count} - WARNING! frame.samples ({frame.samples}) != audio_data_original.shape[0] ({audio_data_original.shape[0]}) for (N,2) stereo.")
                    processed_mono_audio_chunk = audio_data_original.mean(axis=1).astype(np.int16)
                    print(f"STEREO_DEBUG: Frame {iteration_count} - Stereo (N,2) to mono. Output chunk shape: {processed_mono_audio_chunk.shape}")

                    print(f"recv_processor: Frame {iteration_count} - Stereo (N,2) to mono. Chunk shape: {processed_mono_audio_chunk.shape}")
                else:
                    print(f"recv_processor: Frame {iteration_count} - Unexpected shape for stereo: {audio_data_original.shape}. Skipping frame.")
                    continue
            elif current_frame_layout_str == "<av.AudioLayout 'mono'>":
                processed_mono_audio_chunk = audio_data_original.flatten().astype(np.int16) # Гарантируем 1D и int16
                print(f"recv_processor: Frame {iteration_count} - Mono input. Chunk shape: {processed_mono_audio_chunk.shape}")
            else:
                print(f"recv_processor: Frame {iteration_count} - Unknown or unsupported layout: {current_frame_layout_str}. Skipping frame.")
                continue

            if processed_mono_audio_chunk is None or len(processed_mono_audio_chunk) == 0:
                print(f"recv_processor: Frame {iteration_count} - No audio data after mono conversion. Skipping.")
                continue
            
            if processed_mono_audio_chunk is not None and len(processed_mono_audio_chunk) > 0: all_original_mono_chunks_for_debug.append(processed_mono_audio_chunk.copy())
            # Передискретизация (Resampling)
            resampled_audio_chunk = None
            if input_sample_rate != target_sample_rate_vosk:
                num_original_samples = len(processed_mono_audio_chunk)
                num_target_samples = int(num_original_samples * target_sample_rate_vosk / input_sample_rate)
                print(f"recv_processor: Frame {iteration_count} - Resampling: original_mono_samples={num_original_samples} (SR {input_sample_rate}), target_samples={num_target_samples} (SR {target_sample_rate_vosk})")
                if num_target_samples > 0:
                    resampled_audio_chunk = resample(processed_mono_audio_chunk, num_target_samples).astype(np.int16)
                else:
                    resampled_audio_chunk = np.array([], dtype=np.int16)
            else:
                resampled_audio_chunk = processed_mono_audio_chunk # Уже в нужной частоте
                print(f"recv_processor: Frame {iteration_count} - No resampling needed.")

            if len(resampled_audio_chunk) > 0:
                combined_audio_processed.append(resampled_audio_chunk)
                total_processed_samples += len(resampled_audio_chunk)
                print(f"recv_processor: Frame {iteration_count} - Appended {len(resampled_audio_chunk)} resampled samples. Total processed: {total_processed_samples}")

            current_duration_processed = total_processed_samples / target_sample_rate_vosk if target_sample_rate_vosk > 0 else 0
            print(f"recv_processor: Frame {iteration_count} - Current processed duration: {current_duration_processed:.2f}s / {target_duration_seconds}s")
            if current_duration_processed >= target_duration_seconds:
                print(f"recv_processor: Target duration {target_duration_seconds}s reached. Breaking loop.")
                break


        if not combined_audio_processed:
            print("recv_processor: No audio data was processed to combine. Returning None.")
            return None
        if all_original_mono_chunks_for_debug: 
            # concatenated_mono_at_input_sr = np.concatenate(all_original_mono_chunks_for_debug) 
            # target_sample_rate = input_sample_rate/2
            # num_samples = int(len(concatenated_mono_at_input_sr) * (target_sample_rate / input_sample_rate))
            # concatenated_mono_at_input_sr = resample(concatenated_mono_at_input_sr, num_samples)
            concatenated_mono_at_input_sr= np.concatenate(nframe) 
            target_sample_rate = input_sample_rate/2
            num_samples = int(len(concatenated_mono_at_input_sr) * (target_sample_rate / input_sample_rate))
            concatenated_mono_at_input_sr = resample(concatenated_mono_at_input_sr, num_samples)
            print(f"DEBUG: Shape of concatenated_mono_at_input_sr: {concatenated_mono_at_input_sr.shape}") 
            if len(concatenated_mono_at_input_sr) > 0 and input_sample_rate is not None:
                try:
                    with wave.open('debug_FULL_mono_at_48000Hz.wav', 'wb') as wf:
                        wf.setnchannels(2)
                        wf.setsampwidth(2) # int16
                        wf.setframerate(target_sample_rate) # 48000 Гц
                        wf.writeframesraw(concatenated_mono_at_input_sr.astype(np.int16).tobytes())
                    print(f"DEBUG: Saved debug_FULL_mono_at_48000Hz.wav (SR={input_sample_rate})")
                except Exception as e:
                    print(f"DEBUG: Error saving debug_FULL_mono_at_48000Hz.wav: {e}")
            elif input_sample_rate is None:
                print("DEBUG: input_sample_rate is None, cannot save debug_FULL_mono_at_48000Hz.wav")

        final_processed_audio = np.concatenate(combined_audio_processed)
        print(f"recv_processor: Concatenated all processed chunks. Final shape: {final_processed_audio.shape}")

        # Создание AudioFrame для Vosk
        combined_frame_for_vosk = AudioFrame(
            format=target_format_vosk,    # 's16'
            layout=target_layout_vosk,    # 'mono'
            samples=len(final_processed_audio) # Фактическое количество сэмплов в final_processed_audio
        )
        combined_frame_for_vosk.sample_rate = target_sample_rate_vosk
        bytes_to_write = final_processed_audio.tobytes()
        expected_bytes = combined_frame_for_vosk.planes[0].buffer_size

        print(f"recv_processor: For combined_frame_for_vosk - Bytes to write: {len(bytes_to_write)}, Expected plane buffer size: {expected_bytes}")

        if len(bytes_to_write) == expected_bytes:
            combined_frame_for_vosk.planes[0].update(bytes_to_write)
        elif len(bytes_to_write) > expected_bytes:
            print(f"recv_processor: Warning: Truncating data for Vosk AudioFrame plane. Have {len(bytes_to_write)}, need {expected_bytes}.")
            combined_frame_for_vosk.planes[0].update(bytes_to_write[:expected_bytes])
        else: # len(bytes_to_write) < expected_bytes
            # Это более серьезная проблема, т.к. в буфере будут "мусорные" байты
            print(f"recv_processor: ERROR: Not enough bytes for Vosk AudioFrame plane. Have {len(bytes_to_write)}, need {expected_bytes}. Padding with zeros, but this is likely an error in sample counting.")
            padded_bytes = np.zeros(expected_bytes // combined_frame_for_vosk.format.bytes, dtype=np.int16) # //2 for s16
            padded_bytes[:len(final_processed_audio)] = final_processed_audio
            combined_frame_for_vosk.planes[0].update(padded_bytes.tobytes()[:expected_bytes])
        combined_frame_for_vosk.time_base = Fraction(1, target_sample_rate_vosk)
        combined_frame_for_vosk.pts = len(final_processed_audio) # PTS обычно это количество сэмплов с начала потока/сегмента

        print(f"recv_processor: Combined frame for Vosk created: SR={combined_frame_for_vosk.sample_rate}, Samples={combined_frame_for_vosk.samples}, Layout={combined_frame_for_vosk.layout}, PTS={combined_frame_for_vosk.pts}")

        print("recv_processor: Calling STT process_frame...")
        text_result = await self.stt.process_frame(combined_frame_for_vosk)
        print(f"recv_processor: Recognized text from STT: {text_result}")

        return combined_frame_for_vosk # Или text_result, в зависимости от того, что нужно вернуть

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "cv":
            img = frame.to_ndarray(format="bgr24")
            face = faces.detectMultiScale(img, 1.1, 19)
            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            eye = eyes.detectMultiScale(img, 1.1, 19)
            for (x, y, w, h) in eye:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # smile = smiles.detectMultiScale(img, 1.1, 19)
            # for (x, y, w, h) in smile:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 5), 2)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame

def create_local_tracks(play_from=None):
    global relay, webcam

    if play_from:
        player = MediaPlayer(play_from)
        return player.audio, player.video
    else:
        options = {"framerate": "30", "video_size": "640x480"}
        if relay is None:
            print()
            if platform.system() == "Darwin":
                webcam = MediaPlayer(
                    "default:none", format="avfoundation", options=options
                )
            elif platform.system() == "Windows":
               webcam =  MediaPlayer('video=Integrated Camera', format='dshow', options=options)
            elif platform.system() == "Linux":
                webcam =  MediaPlayer('/dev/video0', format='v4l2', options=options)
            relay = MediaRelay()
        return None, relay.subscribe(webcam.video)


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/cv", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_cv.html", {"request": request})

@app.post("/offer")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio, video = create_local_tracks()

    

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        if t.kind == "audio" and audio:
            pc.addTrack(audio)
        elif t.kind == "video" and video:
            pc.addTrack(video)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.post("/offer_cv")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    relay = MediaRelay()
    audio_track_added = False
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # audio, video = create_local_tracks()
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            # transform_audio = await fun(track)\
            pc.addTrack(AudioTransfromTrack(relay.subscribe(track), stt=stt))
        if track.kind == "video":
            transform_video = VideoTransformTrack(relay.subscribe(track), transform=params.video_transform)
            pc.addTrack(transform_video)
        recorder.addTrack(track)
            # if args.record_to:
            #     recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            await recorder.stop()
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        nonlocal audio_track_added
        if pc.iceConnectionState in ["connected", "completed"] and not audio_track_added: # проверка флага
            # Получаем аудио-трек клиента.  
            # Так как фронтенд отправляет и аудио и видео, 
            # то к моменту срабатывания on_iceconnectionstatechange 
            # трек уже должен быть доступен.
            audio_track = next((track for track in pc.getReceivers() if track.track.kind == "audio"), None)
            if audio_track:
                 pc.addTrack(relay.subscribe(audio_track.track))
                 audio_track_added = True # Устанавливаем флаг, чтобы трек не добавлялся повторно
            else:
                print("Аудио трек не найден")
    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


pcs = set()
args = ''


@app.on_event("shutdown")
async def on_shutdown():
    # close peer connections
    print("stop")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="WebRTC webcam demo")
#     parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
#     parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
#     parser.add_argument("--play-from", help="Read the media from a file and sent it.")
#     parser.add_argument(
#         "--play-without-decoding",
#         help=(
#             "Read the media without decoding it (experimental). "
#             "For now it only works with an MPEGTS container with only H.264 video."
#         ),
#         action="store_true",
#     )
#     parser.add_argument(
#         "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
#     )
#     parser.add_argument(
#         "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
#     )
#     parser.add_argument("--verbose", "-v", action="count")
#     parser.add_argument(
#         "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
#     )
#     parser.add_argument(
#         "--video-codec", help="Force a specific video codec (e.g. video/H264)"
#     )

#     args = parser.parse_args()

#     if args.verbose:
#         logging.basicConfig(level=logging.DEBUG)
#     else:
#         logging.basicConfig(level=logging.INFO)

#     if args.cert_file:
#         ssl_context = ssl.SSLContext()
#         ssl_context.load_cert_chain(args.cert_file, args.key_file)
#     else:
#         ssl_context = None

    # app = web.Application()
    # app.on_shutdown.append(on_shutdown)
    # app.router.add_get("/", index)
    # app.router.add_post("/offer", offer)
    # web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)