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
        probabilities = torch.softmax(logits, dim=-1)
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

    # Calculate evenness and distinctness metrics
    syllable_stats = {}
    for syllable in ['p', 't', 'k']:
        syllable_times = [offset for offset in syllable_offsets if offset['char'] == syllable]
        
        if len(syllable_times) > 1:
            intervals = [(syllable_times[i+1]['start_offset'] - syllable_times[i]['start_offset']) * 0.02 
                        for i in range(len(syllable_times)-1)]
            
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = (std_interval / mean_interval) if mean_interval > 0 else 0
            
            # Debug prints for confidence calculation
            syllable_idx = processor.tokenizer.convert_tokens_to_ids(syllable)
            print(f"\nProcessing syllable: {syllable} (token_id: {syllable_idx})")
            confidence_scores = []
            
            # Only look at time windows where this syllable occurs
            for offset in syllable_times:
                # Convert time offset to model timestep index
                time_idx = int(offset['start_offset'])
                prob = probabilities[0][time_idx]
                
                # Get top 5 predictions and their indices
                top_k_values, top_k_indices = torch.topk(prob, k=5)
                
                print(f"\nTimestep {time_idx} (time: {time_idx * 0.02:.3f}s):")
                print(f"Top-5 indices: {top_k_indices.tolist()}")
                print(f"Top-5 values: {top_k_values.tolist()}")
                
                if syllable_idx in top_k_indices:
                    syllable_prob = prob[syllable_idx]
                    relative_confidence = syllable_prob / top_k_values.sum()
                    print(f"Syllable probability: {syllable_prob:.4f}")
                    print(f"Relative confidence: {relative_confidence:.4f}")
                    confidence_scores.append(float(relative_confidence))
                else:
                    confidence_scores.append(0.0)
                    print("Syllable not in top-5")
            
            # Calculate mean confidence only from timesteps where syllable occurs
            mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            print(f"\nFinal confidence scores for {syllable}:")
            print(f"Scores at syllable timestamps: {confidence_scores}")
            print(f"Mean confidence: {mean_confidence:.4f}")
            
            syllable_stats[syllable] = {
                'count': len(syllable_times),
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'cv': cv,
                'mean_confidence': mean_confidence,
                'intervals': intervals,
                'confidence_scores': confidence_scores
            }

    # Create visualization for evenness and distinctness
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color scheme
    colors = {
        'p': '#2E86C1',  # Blue
        't': '#28B463',  # Green
        'k': '#E74C3C'   # Red
    }
    
    # Plot 1: Evenness Analysis
    for syllable, stats in syllable_stats.items():
        if len(stats['intervals']) > 0:
            # Calculate normalized intervals (deviation from mean)
            mean_interval = stats['mean_interval']
            normalized_intervals = [(interval - mean_interval) / mean_interval * 100 
                                 for interval in stats['intervals']]
            
            # Plot normalized intervals
            x = range(len(normalized_intervals))
            ax1.plot(x, normalized_intervals, 'o-', 
                    label=f'{syllable} (CV={stats["cv"]:.2f})',
                    color=colors[syllable], linewidth=2, markersize=8)
            
            # Add individual point annotations
            for i, val in enumerate(normalized_intervals):
                ax1.annotate(f'{val:.1f}%', 
                           (i, val),
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center',
                           fontsize=8)
    
    # Add reference zones for evenness
    ax1.axhspan(-10, 10, color='#2ECC71', alpha=0.2, label='Highly Regular (±10%)')
    ax1.axhspan(-30, -10, color='#F1C40F', alpha=0.2, label='Moderately Regular')
    ax1.axhspan(10, 30, color='#F1C40F', alpha=0.2)
    ax1.axhspan(-50, -30, color='#E74C3C', alpha=0.2, label='Irregular')
    ax1.axhspan(30, 50, color='#E74C3C', alpha=0.2)
    
    ax1.set_xlabel('Repetition Number', fontsize=12)
    ax1.set_ylabel('Deviation from Mean Interval (%)', fontsize=12)
    ax1.set_title('Timing Evenness Analysis\n(Deviation from Mean Interval)', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax1.set_ylim(-50, 50)

    # Plot 2: Distinctness Analysis
    for syllable, stats in syllable_stats.items():
        if len(stats['confidence_scores']) > 0:
            x = range(len(stats['confidence_scores']))
            
            # Create gradient colors based on confidence scores
            colors_array = []
            for score in stats['confidence_scores']:
                if score > 0.7:
                    colors_array.append('#2ECC71')  # Green for high confidence
                elif score > 0.4:
                    colors_array.append('#F1C40F')  # Yellow for medium confidence
                else:
                    colors_array.append('#E74C3C')  # Red for low confidence
            
            # Plot bars with gradient colors
            bars = ax2.bar(x, stats['confidence_scores'], 
                         label=f'{syllable} (mean={stats["mean_confidence"]:.2f})',
                         color=colors_array, alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Add reference lines for distinctness
    ax2.axhline(y=0.7, color='#2ECC71', linestyle='--', alpha=0.5, label='High Distinctness')
    ax2.axhline(y=0.4, color='#F1C40F', linestyle='--', alpha=0.5, label='Moderate Distinctness')
    
    ax2.set_xlabel('Syllable Occurrence', fontsize=12)
    ax2.set_ylabel('Articulation Distinctness Score', fontsize=12)
    ax2.set_title('Articulation Distinctness Analysis\n(Higher Score = Clearer Articulation)', fontsize=14, pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax2.set_ylim(0, 1)

    # Overall layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legends
    plt.savefig('articulation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Update results text with new metrics
    results_text = f"""Syllables per Second Analysis
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SPEED MEASUREMENTS
----------------
- Overall syllables per second: {syllables_per_second:.2f}
- Total number of syllables: {syllable_count}
- Total duration: {audio_duration:.2f} seconds

Detailed Analysis by Syllable:"""

    for syllable, stats in syllable_stats.items():
        results_text += f"""

{syllable.upper()} Syllable Analysis:
Count: {stats['count']} occurrences

EVENNESS MEASUREMENTS (timing regularity)
--------------------------------
- Mean interval between repetitions: {stats['mean_interval']:.3f} seconds
- Variation in intervals (std dev): {stats['std_interval']:.3f} seconds
- Coefficient of variation: {stats['cv']:.3f}
  (Lower CV = more even timing, Higher CV = more irregular timing)
  * CV < 0.1: Highly regular
  * CV 0.1-0.3: Moderately regular
  * CV > 0.3: Irregular

DISTINCTNESS MEASUREMENTS (articulation clarity)
------------------------------------
- Mean articulation confidence: {stats['mean_confidence']:.3f}
  (Higher values indicate clearer articulation)
  * Values closer to 1.0 indicate very distinct pronunciation
  * Values closer to 0.0 indicate less distinct pronunciation
- Confidence scores for each occurrence: {stats['confidence_scores']}

RAW MEASUREMENTS
--------------
- All intervals between repetitions (seconds): {stats['intervals']}"""

    # Print the results text to verify
    print("\nFinal Results Text:")
    print(results_text)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the audio file
    audio_path = 'results/recorded_audio.wav'
    with open(audio_path, 'wb') as f:
        f.write(audio_bytes)
    
    # Save syllables per second to text file
    with open('results/analysis_results.txt', 'w') as f:
        f.write(results_text)
    
    # Create zip file
    zip_path = 'results/analysis_package.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write('syllables_per_second.png')
        zipf.write('mel_spectrogram.png')
        zipf.write('articulation_analysis.png')
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

    # Display all visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.image('syllables_per_second.png')
        st.image('articulation_analysis.png')
    with col2:
        st.image('mel_spectrogram.png')

    # Display detailed metrics
    st.write("### Detailed Analysis")
    for syllable, stats in syllable_stats.items():
        st.write(f"\n**{syllable.upper()} Syllable:**")
        st.write(f"- Count: {stats['count']}")
        st.write(f"- Mean interval: {stats['mean_interval']:.3f} seconds")
        st.write(f"- Coefficient of variation: {stats['cv']:.3f}")
        st.write(f"- Mean articulation confidence: {stats['mean_confidence']:.3f}")

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
