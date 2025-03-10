import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
import soundfile as sf
import io
import librosa
import matplotlib.pyplot as plt
import librosa.display
import zipfile
import os
from datetime import datetime

# For recording
from st_audiorec import st_audiorec
import base64
from pydub import AudioSegment

st.title("Syllables per Second Calculator")
st.write(
    "Upload an audio file *or* record from your microphone to calculate "
    "the number of 'p', 't', and 'k' syllables per second."
)

def get_syllables_per_second(audio_bytes):
    """
    Processes an audio file-like object (or BytesIO) and returns syllables/sec.
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    # Read the audio data
    audio_input, original_sample_rate = sf.read(io.BytesIO(audio_bytes))
    target_sample_rate = processor.feature_extractor.sampling_rate

    # Resample if needed
    if original_sample_rate != target_sample_rate:
        if audio_input.ndim > 1:
            audio_input = np.asarray([
                librosa.resample(
                    channel,
                    orig_sr=original_sample_rate,
                    target_sr=target_sample_rate
                ) 
                for channel in audio_input.T
            ]).T
        else:
            audio_input = librosa.resample(
                audio_input,
                orig_sr=original_sample_rate,
                target_sr=target_sample_rate
            )

    # Convert to mono if stereo
    if audio_input.ndim > 1:
        audio_input = np.mean(audio_input, axis=1)

    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_input,
        sr=target_sample_rate,
        n_mels=128,
        fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Prepare input for the model
    input_values = processor(audio_input, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, output_char_offsets=True)
        offsets = transcription["char_offsets"]
        print("Offsets are:", offsets)

    # We only care about 'p', 't', or 'k'
    syllable_offsets = [item for item in offsets[0] if item['char'] in ['p', 't', 'k']]
    
    if syllable_offsets:
        first_syllable_offset = syllable_offsets[0]['start_offset'] * 0.02
        last_syllable_offset = syllable_offsets[-1]['end_offset'] * 0.02
        syllable_duration = last_syllable_offset - first_syllable_offset
    else:
        syllable_duration = 0

    syllable_count = len(syllable_offsets)
    audio_duration = len(audio_input) / target_sample_rate
    syllables_per_second = syllable_count / syllable_duration if syllable_duration > 0 else 0

    # Plot syllables per second over time
    times = []
    syllables_per_second_time = []
    for i in range(len(syllable_offsets) - 1):
        start = syllable_offsets[i]['start_offset'] * 0.02
        end = syllable_offsets[i + 1]['end_offset'] * 0.02
        duration = end - start
        rate = 1 / duration if duration > 0 else 0
        times.append(start)
        syllables_per_second_time.append(rate)

    plt.figure(figsize=(8, 3))
    plt.plot(times, syllables_per_second_time, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Syllables per second')
    plt.title('Syllables/Second Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('syllables_per_second.png')
    plt.close()

    # Create a new figure for the mel spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        mel_spec_db,
        sr=target_sample_rate,
        x_axis='time',
        y_axis='mel',
        fmax=8000
    )
    
    # Highlight p, t, k sounds
    for offset in syllable_offsets:
        start_time = offset['start_offset'] * 0.02
        end_time = offset['end_offset'] * 0.02
        mid_time = (start_time + end_time) / 2
        plt.axvline(x=start_time, color='r', alpha=0.3, linestyle='--')
        plt.text(mid_time, mel_spec_db.shape[0] * 0.9, 
                offset['char'].upper(),
                horizontalalignment='center',
                color='white',
                bbox=dict(facecolor='red', alpha=0.7))

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram with Highlighted Syllables')
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png')
    plt.close()

    # Display both visualizations side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image('syllables_per_second.png')
    with col2:
        st.image('mel_spectrogram.png')

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the audio file
    audio_path = 'results/recorded_audio.wav'
    with open(audio_path, 'wb') as f:
        f.write(audio_bytes)
    
    # Save syllables per second to text file
    results_text = f"""Syllables per Second Analysis
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Syllables per second: {syllables_per_second:.2f}
Number of syllables: {syllable_count}
Total duration: {audio_duration:.2f} seconds
"""
    with open('results/analysis_results.txt', 'w') as f:
        f.write(results_text)
    
    # Create zip file
    zip_path = 'results/analysis_package.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write('syllables_per_second.png')
        zipf.write('mel_spectrogram.png')
        zipf.write(audio_path)
        zipf.write('results/analysis_results.txt')
    
    # Add download button after the visualizations
    with open(zip_path, 'rb') as f:
        st.download_button(
            label="Download Analysis Package",
            data=f,
            file_name="ddk_analysis_package.zip",
            mime="application/zip",
            help="Download a zip file containing the audio, visualizations, and analysis results"
        )

    return syllables_per_second


# -----------------------------
# SECTION: File Uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    with st.spinner("Processing the uploaded audio file..."):
        # Read entire file into bytes
        audio_bytes = uploaded_file.read()
        result = get_syllables_per_second(audio_bytes)
        st.write(f"**Syllables per second (uploaded):** {result:.2f}")


# -----------------------------
# SECTION: Audio Recorder
# -----------------------------
st.write("---")
st.subheader("Or record audio from your microphone")

# The st_audiorec component returns base64 encoded wav data
recorded_data = st_audiorec()

if recorded_data is not None:
    st.info("Audio recording complete. Processing ...")
    
    # Check if recorded_data is bytes or string
    if isinstance(recorded_data, bytes):
        decoded = recorded_data
    else:
        # Convert the base64 encoded data to wav audio
        # recorded_data is a base64 string with headers, so we split off the prefix
        try:
            header, encoded = recorded_data.split(",", 1)
            decoded = base64.b64decode(encoded)
        except AttributeError:
            st.error("Unexpected audio format received from recorder")
            st.stop()

    # Rest of the processing remains the same
    audio_segment = AudioSegment.from_file(io.BytesIO(decoded), format="wav")
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_bytes = wav_io.getvalue()

    # Run the analysis using the same function
    with st.spinner("Analyzing your recorded audio..."):
        recorded_result = get_syllables_per_second(wav_bytes)
        st.write(f"**Syllables per second (recorded):** {recorded_result:.2f}")
