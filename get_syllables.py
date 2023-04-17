from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
import soundfile as sf

def get_syllables_per_second(audio_file):
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    # load dataset and read soundfiles
    audio_input, sample_rate = sf.read(audio_file)

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
        offsets = transcription['char_offsets']

    # Calculate syllables per second
    audio_duration = len(audio_input) / sample_rate
    syllable_count = sum(1 for item in offsets[0] if item['char'] in ['p', 't', 'k'])
    
    syllables_per_second = syllable_count / audio_duration

    return syllables_per_second

# Example usage
audio_file = "pataka_josefin.WAV"
syllables_per_second = get_syllables_per_second(audio_file)
print("Syllables per second:", syllables_per_second)
audio_file = "pataka_16k.wav"
syllables_per_second = get_syllables_per_second(audio_file)
print("Syllables per second:", syllables_per_second)