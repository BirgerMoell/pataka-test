import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
import soundfile as sf
import io

st.title("Syllables per Second Calculator")
st.write("Upload an audio file to calculate the number of 'p', 't', and 'k' syllables per second.")

def get_syllables_per_second(audio_file):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    audio_input, sample_rate = sf.read(io.BytesIO(audio_file.read()))

    if audio_input.ndim > 1 and audio_input.shape[1] == 2:
        audio_input = np.mean(audio_input, axis=1)

    input_values = processor(audio_input, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, output_char_offsets=True)
        offsets = transcription['char_offsets']

    # Find the start and end time offsets of the syllables
    syllable_offsets = [item for item in offsets[0] if item['char'] in ['p', 't', 'k']]
    
    if syllable_offsets:  # if any syllable is found
        first_syllable_offset = syllable_offsets[0]['start_offset'] / sample_rate
        last_syllable_offset = syllable_offsets[-1]['end_offset'] / sample_rate
        # Duration from the first to the last syllable
        syllable_duration = last_syllable_offset - first_syllable_offset
    else:
        syllable_duration = 0

    syllable_count = len(syllable_offsets)
    syllables_per_second = syllable_count / syllable_duration if syllable_duration > 0 else 0

    return syllables_per_second

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Processing the audio file..."):
        result = get_syllables_per_second(uploaded_file)
        st.write("Syllables per second: ", result)
