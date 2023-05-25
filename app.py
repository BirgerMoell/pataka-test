import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
import soundfile as sf
import io
import librosa
import matplotlib.pyplot as plt


st.title("Syllables per Second Calculator")
st.write("Upload an audio file to calculate the number of 'p', 't', and 'k' syllables per second.")

def get_syllables_per_second(audio_file):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    audio_input, original_sample_rate = sf.read(io.BytesIO(audio_file.read()))
    target_sample_rate = processor.feature_extractor.sampling_rate

    # resample the sample rate if not 16 k
    if original_sample_rate != target_sample_rate:
        if audio_input.ndim > 1:
            audio_input = np.asarray([librosa.resample(channel, orig_sr=original_sample_rate, target_sr=target_sample_rate) for channel in audio_input.T]).T
        else:
            audio_input = librosa.resample(audio_input, orig_sr=original_sample_rate, target_sr=target_sample_rate)

    # make the audio mono if it is stereo
    if audio_input.ndim > 1 and audio_input.shape[1] == 2:
        audio_input = np.mean(audio_input, axis=1)

    input_values = processor(audio_input, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, output_char_offsets=True)
        offsets = transcription['char_offsets']
        print("the offets are: ", offsets)

    # Find the start and end time offsets of the syllables

    syllable_offsets = [item for item in offsets[0] if item['char'] in ['p', 't', 'k']]
    
    if syllable_offsets:  # if any syllable is found
        first_syllable_offset = syllable_offsets[0]['start_offset'] * 0.02
        last_syllable_offset = syllable_offsets[-1]['end_offset'] * 0.02

        print("the first syllable offset is: ", first_syllable_offset)
        print("the last syllable offset is: ", last_syllable_offset)
        # Duration from the first to the last syllable
        syllable_duration = last_syllable_offset - first_syllable_offset
        print("the syllable duration is: ", syllable_duration)
    else:
        syllable_duration = 0

    syllable_count = len(syllable_offsets)
    audio_duration = len(audio_input) / target_sample_rate
    print("the audio duration is: ", audio_duration)
    print("the syllable count is: ", syllable_count)
    #print("the syllabels per second is: ", syllable_count / audio_duration)
    syllables_per_second = syllable_count / syllable_duration if syllable_duration > 0 else 0

    times = []
    syllables_per_second_time = []
    for i in range(len(syllable_offsets) - 1):
        start = syllable_offsets[i]['start_offset'] * 0.02
        end = syllable_offsets[i + 1]['end_offset'] * 0.02
        duration = end - start
        rate = 1 / duration if duration > 0 else 0
        times.append(start)
        syllables_per_second_time.append(rate)

    plt.plot(times, syllables_per_second_time)
    plt.xlabel('Time (s)')
    plt.ylabel('Syllables per second')
    # plt.show()
    # save the figure
    plt.savefig('syllables_per_second.png')
    # show the image using streamlit
    st.image('syllables_per_second.png')

    return syllables_per_second

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Processing the audio file..."):
        result = get_syllables_per_second(uploaded_file)
        st.write("Syllables per second: ", result)