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
    offsets = transcription['char_offsets']

characters = [[{'char': 'p', 'start_offset': 64, 'end_offset': 65}, {'char': 'a', 'start_offset': 66, 'end_offset': 67}, {'char': 't', 'start_offset': 69, 'end_offset': 70}, {'char': 'ɛ', 'start_offset': 71, 'end_offset': 72}, {'char': 'k', 'start_offset': 74, 'end_offset': 75}, {'char': 'ɛ', 'start_offset': 76, 'end_offset': 77}, {'char': 'p', 'start_offset': 79, 'end_offset': 80}, {'char': 'a', 'start_offset': 80, 'end_offset': 81}, {'char': 't', 'start_offset': 83, 'end_offset': 85}, {'char': 'ɛ', 'start_offset': 85, 'end_offset': 86}, {'char': 'k', 'start_offset': 88, 'end_offset': 89}, {'char': 'ɛ', 'start_offset': 90, 'end_offset': 91}, {'char': 'p', 'start_offset': 93, 'end_offset': 94}, {'char': 'ɑ', 'start_offset': 95, 'end_offset': 96}, {'char': 't', 'start_offset': 99, 'end_offset': 100}, {'char': 'ɛ', 'start_offset': 100, 'end_offset': 101}, {'char': 'k', 'start_offset': 103, 'end_offset': 104}, {'char': 'ɛ', 'start_offset': 105, 'end_offset': 106}, {'char': 'p', 'start_offset': 108, 'end_offset': 109}, {'char': 'ɑ', 'start_offset': 110, 'end_offset': 111}, {'char': 't', 'start_offset': 114, 'end_offset': 115}, {'char': 'ɛ', 'start_offset': 115, 'end_offset': 116}, {'char': 'k', 'start_offset': 118, 'end_offset': 119}, {'char': 'ɛ', 'start_offset': 120, 'end_offset': 121}, {'char': 'p', 'start_offset': 123, 'end_offset': 124}, {'char': 'ɑ', 'start_offset': 125, 'end_offset': 126}, {'char': 't', 'start_offset': 129, 'end_offset': 130}, {'char': 'k', 'start_offset': 133, 'end_offset': 134}, {'char': 'p', 'start_offset': 138, 'end_offset': 139}, {'char': 't', 'start_offset': 143, 'end_offset': 144}, {'char': 'k', 'start_offset': 149, 'end_offset': 150}, {'char': 'ɛ', 'start_offset': 150, 'end_offset': 151}, {'char': 'p', 'start_offset': 154, 'end_offset': 155}, {'char': 't', 'start_offset': 159, 'end_offset': 160}, {'char': 'ɛ', 'start_offset': 161, 'end_offset': 162}, {'char': 'k', 'start_offset': 164, 'end_offset': 165}, {'char': 'ɛ', 'start_offset': 166, 'end_offset': 167}, {'char': 'p', 'start_offset': 169, 'end_offset': 170}, {'char': 'ɑ', 'start_offset': 171, 'end_offset': 172}, {'char': 't', 'start_offset': 174, 'end_offset': 175}, {'char': 'ɛ', 'start_offset': 176, 'end_offset': 177}, {'char': 'k', 'start_offset': 179, 'end_offset': 180}, {'char': 'ɛ', 'start_offset': 181, 'end_offset': 182}, {'char': 'p', 'start_offset': 184, 'end_offset': 185}, {'char': 'ɑ', 'start_offset': 186, 'end_offset': 187}, {'char': 't', 'start_offset': 189, 'end_offset': 190}, {'char': 'ɛ', 'start_offset': 191, 'end_offset': 192}, {'char': 'k', 'start_offset': 194, 'end_offset': 195}, {'char': 'ɛ', 'start_offset': 196, 'end_offset': 197}, {'char': 'p', 'start_offset': 200, 'end_offset': 201}, {'char': 'a', 'start_offset': 201, 'end_offset': 202}, {'char': 't', 'start_offset': 205, 'end_offset': 206}, {'char': 'k', 'start_offset': 210, 'end_offset': 211}, {'char': 'ɛ', 'start_offset': 212, 'end_offset': 213}, {'char': 'p', 'start_offset': 215, 'end_offset': 216}, {'char': 'a', 'start_offset': 217, 'end_offset': 218}, {'char': 't', 'start_offset': 220, 'end_offset': 221}, {'char': 'ɛ', 'start_offset': 222, 'end_offset': 223}, {'char': 'k', 'start_offset': 225, 'end_offset': 226}, {'char': 'ɛ', 'start_offset': 227, 'end_offset': 228}, {'char': 'p', 'start_offset': 230, 'end_offset': 231}, {'char': 'a', 'start_offset': 232, 'end_offset': 233}, {'char': 't', 'start_offset': 236, 'end_offset': 237}, {'char': 'ɛ', 'start_offset': 238, 'end_offset': 239}, {'char': 'k', 'start_offset': 242, 'end_offset': 243}, {'char': 'ɛ', 'start_offset': 244, 'end_offset': 245}, {'char': 'p', 'start_offset': 247, 'end_offset': 248}, {'char': 'a', 'start_offset': 249, 'end_offset': 250}, {'char': 't', 'start_offset': 253, 'end_offset': 254}, {'char': 'a', 'start_offset': 255, 'end_offset': 256}, {'char': 'k', 'start_offset': 258, 'end_offset': 259}]]

# Calculate syllables per second
audio_duration = len(audio_input) / sample_rate
#syllable_count = sum(1 for item in offsets[0] if item['char'] != ' ')
syllable_count = sum(1 for item in offsets[0] if item['char'] in ['p', 't', 'k'])

print("there are", syllable_count, "syllables in the audio file")
print("the audio file is", audio_duration, "seconds long")
syllables_per_second = syllable_count / audio_duration

print("Syllables per second:", syllables_per_second)