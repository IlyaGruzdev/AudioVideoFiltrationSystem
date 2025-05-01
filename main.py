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
from av import VideoFrame
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

faces = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
smiles = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")

class AudioTransfromTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, track, stt: STT):
        super().__init__()
        self.track = track
        self.stt = stt

    async def recv(self):
        frame = await self.track.recv()
        text = await self.stt.process_frame(frame)
        if text:
            print("Распознанный текст:", text)
        return frame

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
    stt = STT(modelpath="voice_recognition/vosk-model-ru-0.42", sample_rate=16000)

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