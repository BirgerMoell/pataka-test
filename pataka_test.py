import time
import os
import sys
import whisper

model = whisper.load_model("large")
result = model.transcribe("pataka.WAV", initial_prompt='pataka pataka pataka pataka pataka pataka')
print(result["text"])