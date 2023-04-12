from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

import numpy as np
import soundfile as sf

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# load dummy dataset and read soundfiles
audio_input, sample_rate = sf.read("pataka_16k.wav")

# Check if audio is stereo (2 channels) and convert to mono if necessary
if audio_input.ndim > 1 and audio_input.shape[1] == 2:
    audio_input = np.mean(audio_input, axis=1)

# tokenize
input_values = processor(audio_input, return_tensors="pt").input_values

# retrieve logits
with torch.no_grad():
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, output_char_offsets=True)
    print(transcription)