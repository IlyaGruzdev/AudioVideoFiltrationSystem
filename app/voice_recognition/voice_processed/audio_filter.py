import sounddevice as sd
import torch
import time


# константы голосов, поддерживаемых в silero
SPEAKER_AIDAR   = "aidar"
SPEAKER_BAYA    = "baya"
SPEAKER_KSENIYA = "kseniya"
SPEAKER_XENIA   = "xenia"
SPEAKER_RANDOM  = "random"

# константы девайсов для работы torch
DEVICE_CPU    = "cpu"
DEVICE_CUDA   = "cuda" 
DEVICE_VULKAN = "vulkan"
DEVICE_OPENGL = "opengl"
DEVICE_OPENCL = "opencl"

class AudioFilter:
  pass