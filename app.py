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
import math
import scipy.signal

# For recording
from st_audiorec import st_audiorec
import base64
from pydub import AudioSegment

# =============================================================================
# AUDIO PROCESSING FUNCTIONS
# =============================================================================

def load_and_preprocess_audio(audio_bytes):
    """Load and preprocess audio data for analysis."""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    
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

    return audio_input, target_sample_rate, processor

def create_mel_spectrogram(audio_input, sample_rate, hop_length):
    """Create ultra high-quality mel spectrogram from audio data."""
    # Use speech-optimized frequency range (0-8000 Hz for better detail in speech range)
    fmax = min(8000, sample_rate//2)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_input,
        sr=sample_rate,
        n_mels=128,  # Doubled for much finer frequency resolution
        hop_length=hop_length,  # Smaller hop for better time resolution
        fmax=fmax,  # Focus on speech-relevant frequencies
        n_fft=512,  # Much larger FFT window for better frequency resolution
        window='hann',  # Explicit window function
        center=True,  # Center the window
        pad_mode='constant'  # Better padding
    )
    # Use percentile-based scaling for better detail in quieter parts
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.percentile(mel_spec, 95))
    return mel_spec_db

def transcribe_audio(audio_input, processor):
    """Transcribe audio and extract syllable offsets."""
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    
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
    
    return syllable_offsets, probabilities

def calculate_syllable_metrics(syllable_offsets, audio_input, sample_rate):
    """Calculate basic syllable metrics."""
    # Wav2Vec2 uses 20ms frames (0.02 seconds) for char offsets
    frame_time = 0.02  # Wav2Vec2 frame rate
    
    if syllable_offsets:
        first_syllable_offset = syllable_offsets[0]['start_offset'] * frame_time
        last_syllable_offset = syllable_offsets[-1]['end_offset'] * frame_time
        syllable_duration = last_syllable_offset - first_syllable_offset
    else:
        syllable_duration = 0

    syllable_count = len(syllable_offsets)
    audio_duration = len(audio_input) / sample_rate
    syllables_per_second = syllable_count / syllable_duration if syllable_duration > 0 else 0

    return syllable_count, audio_duration, syllables_per_second, syllable_duration

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_syllable_stats(syllable_offsets, processor, probabilities, sample_rate):
    """Calculate detailed statistics for each syllable type."""
    # Wav2Vec2 uses 20ms frames (0.02 seconds) for char offsets
    frame_time = 0.02  # Wav2Vec2 frame rate
    
    syllable_stats = {}
    
    for syllable in ['p', 't', 'k']:
        syllable_times = [offset for offset in syllable_offsets if offset['char'] == syllable]
        
        if len(syllable_times) > 1:
            intervals = [(syllable_times[i+1]['start_offset'] - syllable_times[i]['start_offset']) * frame_time 
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
    
    return syllable_stats

def calculate_energy_metrics(audio_input, sample_rate, syllable_offsets):
    """Calculate energy-related metrics for each syllable."""
    energy_stats = {}
    window_size = int(0.02 * sample_rate)  # 20ms window
    
    for syllable in ['p', 't', 'k']:
        syllable_times = [offset for offset in syllable_offsets if offset['char'] == syllable]
        peak_energies = []
        energy_spreads = []
        
        for time in syllable_times:
            start_sample = int(time['start_offset'] * 0.02 * sample_rate)
            end_sample = int(time['end_offset'] * 0.02 * sample_rate)
            
            if end_sample <= len(audio_input):
                # Get syllable segment
                segment = audio_input[start_sample:end_sample]
                
                # Calculate RMS energy
                energy = np.sqrt(np.mean(segment**2))
                peak_energies.append(float(energy))
                
                # Calculate spectral spread
                if len(segment) >= window_size:
                    freqs, times, spec = scipy.signal.spectrogram(
                        segment,
                        fs=sample_rate,
                        nperseg=window_size,
                        noverlap=window_size//2
                    )
                    # Calculate spread of energy across frequencies
                    spread = np.std(np.mean(spec, axis=1))
                    energy_spreads.append(float(spread))
        
        if peak_energies:
            mean_energy = np.mean(peak_energies)
            energy_cv = np.std(peak_energies) / mean_energy if mean_energy > 0 else 0
            
            energy_stats[syllable] = {
                'peak_energies': peak_energies,
                'mean_energy': mean_energy,
                'energy_cv': energy_cv,
                'energy_spreads': energy_spreads
            }
    
    return energy_stats

def add_energy_to_syllable_stats(syllable_stats, audio_input, sample_rate):
    """Add energy metrics to existing syllable statistics."""
    for syllable, stats in syllable_stats.items():
        if len(stats['intervals']) > 0:
            # Calculate energy at each syllable timestamp
            energies = []
            for time in stats['intervals']:
                start_sample = int(time * sample_rate)
                end_sample = int((time + 0.02) * sample_rate)  # 20ms window
                
                if end_sample <= len(audio_input):
                    segment = audio_input[start_sample:end_sample]
                    energy = np.sqrt(np.mean(segment**2))
                    energies.append(float(energy))
            
            # Store energy values in stats
            stats['energies'] = energies
            stats['mean_energy'] = np.mean(energies)
            stats['energy_cv'] = np.std(energies) / stats['mean_energy'] if stats['mean_energy'] > 0 else 0

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_syllables_per_second_plot(syllable_offsets, sample_rate):
    """Create syllables per second over time plot."""
    # Wav2Vec2 uses 20ms frames (0.02 seconds) for char offsets
    frame_time = 0.02  # Wav2Vec2 frame rate
    
    times = []
    syllables_per_second_time = []
    for i in range(len(syllable_offsets) - 1):
        start = syllable_offsets[i]['start_offset'] * frame_time
        end = syllable_offsets[i + 1]['end_offset'] * frame_time
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

def create_mel_spectrogram_plot(mel_spec_db, sample_rate, syllable_offsets, hop_length):
    """Create ultra high-quality mel spectrogram with highlighted syllables."""
    # Use speech-optimized frequency range for better detail
    fmax = min(8000, sample_rate//2)
    
    
    # Wav2Vec2 uses 20ms frames (0.02 seconds) for char offsets
    # This is independent of the mel spectrogram hop_length
    frame_time = 0.02  # Wav2Vec2 frame rate
    
    plt.figure(figsize=(16, 5), dpi=300)  # Much larger figure and ultra-high DPI
    librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        hop_length=hop_length,  # This is crucial for correct time axis scaling
        x_axis='time',
        y_axis='mel',
        fmax=fmax,  # Focus on speech-relevant frequencies
        cmap='Greys'  # High contrast colormap for better readability
    )
    
    # Highlight p, t, k sounds using the correct time conversion
    for offset in syllable_offsets:
        start_time = offset['start_offset'] * frame_time
        end_time = offset['end_offset'] * frame_time
        mid_time = (start_time + end_time) / 2
        plt.axvline(x=start_time, color='r', alpha=0.3, linestyle='--')
    
    # Add labels below the time axis
    # Get the current y-axis limits and extend them downward
    y_min, y_max = plt.ylim()
    # Calculate a fixed offset below the spectrogram for consistent label placement (reduced space)
    label_y_position = y_min - (y_max - y_min) * 0.025  # Position labels closer to spectrogram
    plt.ylim(y_min - (y_max - y_min) * 0.04, y_max)  # Minimal extension for labels (25% of previous)

    for offset in syllable_offsets:
        start_time = offset['start_offset'] * frame_time
        end_time = offset['end_offset'] * frame_time
        mid_time = (start_time + end_time) / 2
        plt.text(mid_time, label_y_position, 
                offset['char'].upper(),
                horizontalalignment='left',
                verticalalignment='center',
                color='white',
                bbox=dict(facecolor='red', alpha=0.8),
                fontsize=14,
                fontweight='bold')
    
    # Hide y-axis labels in the negative range where labels are positioned
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    # Only show positive y-axis labels (above 0)
    ax.set_yticks([tick for tick in y_ticks if tick >= 0])

    #plt.colorbar(format='%+2.0f dB', shrink=0.8)
    plt.title('Mel Spectrogram with Highlighted Syllables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')  # Ultra-high DPI for maximum crispness
    plt.close()

def create_evenness_analysis_plot(syllable_stats):
    """Create evenness analysis visualization."""
    plt.figure(figsize=(15, 8))  # Increased figure size
    
    # Color scheme
    syllable_colors = {
        'p': '#2E86C1',  # Blue
        't': '#28B463',  # Green
        'k': '#E74C3C'   # Red
    }
    
    # Track maximum deviation to set y-axis limits
    max_deviation = 0
    
    # First pass to calculate all points for smart annotation placement
    all_points = {}
    for syllable, stats in syllable_stats.items():
        if len(stats['intervals']) > 0:
            mean_interval = stats['mean_interval']
            normalized_intervals = [(interval - mean_interval) / mean_interval * 100 
                                 for interval in stats['intervals']]
            all_points[syllable] = list(zip(range(len(normalized_intervals)), normalized_intervals))
            
            # Update max deviation
            current_max = max(abs(min(normalized_intervals)), abs(max(normalized_intervals)))
            max_deviation = max(max_deviation, current_max)
    
    # Plot the lines and points
    for syllable, stats in syllable_stats.items():
        if len(stats['intervals']) > 0:
            mean_interval = stats['mean_interval']
            normalized_intervals = [(interval - mean_interval) / mean_interval * 100 
                                 for interval in stats['intervals']]
            
            x = range(len(normalized_intervals))
            plt.plot(x, normalized_intervals, 'o-', 
                    label=f'{syllable} (CV={stats["cv"]:.2f})',
                    color=syllable_colors[syllable], linewidth=2, markersize=8)
            
            # Smart annotation placement
            for i, val in enumerate(normalized_intervals):
                # Find nearby points from other syllables
                nearby_points = []
                for other_syllable, points in all_points.items():
                    if other_syllable != syllable:
                        for px, py in points:
                            if px == i and abs(py - val) < 15:  # 15% threshold for "nearby"
                                nearby_points.append(py)
                
                # Adjust y-offset based on number of nearby points
                y_offset = 10
                if nearby_points:
                    if val > max(nearby_points):
                        y_offset = 20
                    elif val < min(nearby_points):
                        y_offset = -20
                
                plt.annotate(f'{val:.1f}%', 
                           (i, val),
                           xytext=(0, y_offset), 
                           textcoords='offset points',
                           ha='center',
                           va='bottom' if y_offset > 0 else 'top',
                           fontsize=8,
                           color=syllable_colors[syllable])

    # Calculate y-axis limits with 10% padding
    y_limit = max(50, math.ceil(max_deviation * 1.1 / 10) * 10)
    
    # Add reference zones for evenness (extend to calculated limits)
    plt.axhspan(-10, 10, color='#2ECC71', alpha=0.2, label='Highly Regular (±10%)')
    plt.axhspan(-30, -10, color='#F1C40F', alpha=0.2, label='Moderately Regular')
    plt.axhspan(10, 30, color='#F1C40F', alpha=0.2)
    plt.axhspan(-y_limit, -30, color='#E74C3C', alpha=0.2, label='Irregular')
    plt.axhspan(30, y_limit, color='#E74C3C', alpha=0.2)
    
    plt.xlabel('Repetition Number', fontsize=12)
    plt.ylabel('Deviation from Mean Interval (%)', fontsize=12)
    plt.title('Timing Evenness Analysis\n(Deviation from Mean Interval)', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    plt.savefig('evenness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distinctness_analysis_plot(syllable_stats):
    """Create distinctness analysis visualization."""
    plt.figure(figsize=(15, 8))
    
    # Color schemes
    syllable_colors = {
        'p': '#2E86C1',  # Blue
        't': '#28B463',  # Green
        'k': '#E74C3C'   # Red
    }
    
    distinctness_colors = {
        'high': '#2ECC71',    # Green
        'moderate': '#F1C40F', # Yellow
        'low': '#E74C3C'      # Red
    }
    
    bar_width = 0.25  # Width of each bar
    for syllable, stats in syllable_stats.items():
        if len(stats['confidence_scores']) > 0:
            x = range(len(stats['confidence_scores']))
            
            # Create bars with colors based on distinctness levels
            for i, (score, time) in enumerate(zip(stats['confidence_scores'], stats['intervals'])):
                if score > 0.7:
                    color = distinctness_colors['high']
                    alpha = 0.9
                elif score > 0.4:
                    color = distinctness_colors['moderate']
                    alpha = 0.7
                else:
                    color = distinctness_colors['low']
                    alpha = 0.5
                
                # Position bars for each syllable
                x_pos = i + {'p': 0, 't': bar_width, 'k': 2*bar_width}[syllable]
                bar = plt.bar(x_pos, score,
                            bar_width,
                            color=syllable_colors[syllable],
                            alpha=alpha,
                            label=syllable if i == 0 else "")
                
                # Add value labels on top of bars
                plt.text(x_pos, score, f'{score:.2f}\n{time:.2f}s',
                        ha='center', va='bottom', fontsize=8)
    
    # Add reference lines for distinctness
    plt.axhline(y=0.7, color=distinctness_colors['high'], linestyle='--', 
                alpha=0.5, label='High Distinctness')
    plt.axhline(y=0.4, color=distinctness_colors['moderate'], linestyle='--', 
                alpha=0.5, label='Moderate Distinctness')
    
    plt.xlabel('Syllable Occurrence (with timestamp)', fontsize=12)
    plt.ylabel('Articulation Distinctness Score', fontsize=12)
    plt.title('Articulation Distinctness Analysis\n(Higher Score = Clearer Articulation)', 
              fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('distinctness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_energy_analysis_plot(syllable_stats, audio_input, sample_rate):
    """Create energy analysis visualization."""
    plt.figure(figsize=(15, 8))
    
    # Color scheme
    syllable_colors = {
        'p': '#2E86C1',  # Blue
        't': '#28B463',  # Green
        'k': '#E74C3C'   # Red
    }
    
    # Track maximum energy deviation
    max_energy_deviation = 0
    
    for syllable, stats in syllable_stats.items():
        if len(stats['intervals']) > 0:
            # Calculate energy at each syllable timestamp
            energies = []
            for time in stats['intervals']:
                start_sample = int(time * sample_rate)
                end_sample = int((time + 0.02) * sample_rate)  # 20ms window
                
                if end_sample <= len(audio_input):
                    segment = audio_input[start_sample:end_sample]
                    energy = np.sqrt(np.mean(segment**2))
                    energies.append(float(energy))
            
            # Store energy values in stats
            stats['energies'] = energies
            stats['mean_energy'] = np.mean(energies)
            stats['energy_cv'] = np.std(energies) / stats['mean_energy'] if stats['mean_energy'] > 0 else 0
            
            # Calculate normalized energies
            normalized_energies = [(e - stats['mean_energy']) / stats['mean_energy'] * 100 
                                 for e in energies]
            
            # Update max deviation
            current_max = max(abs(min(normalized_energies)), abs(max(normalized_energies)))
            max_energy_deviation = max(max_energy_deviation, current_max)
            
            # Plot energy pattern
            x = range(len(normalized_energies))
            plt.plot(x, normalized_energies, 'o-', 
                    label=f'{syllable} (CV={stats["energy_cv"]:.2f})',
                    color=syllable_colors[syllable], linewidth=2, markersize=8)
            
            # Add timing annotations
            for i, (energy, time) in enumerate(zip(normalized_energies, stats['intervals'])):
                plt.annotate(f'{time:.2f}s', 
                           (i, energy),
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center',
                           rotation=45,
                           fontsize=8,
                           color=syllable_colors[syllable])
    
    # Calculate y-axis limits with 10% padding
    y_limit = max(50, math.ceil(max_energy_deviation * 1.1 / 10) * 10)
    
    # Add reference zones
    plt.axhspan(-15, 15, color='#2ECC71', alpha=0.2, label='Consistent Energy (±15%)')
    plt.axhspan(-30, -15, color='#F1C40F', alpha=0.2, label='Moderate Variation')
    plt.axhspan(15, 30, color='#F1C40F', alpha=0.2)
    plt.axhspan(-y_limit, -30, color='#E74C3C', alpha=0.2, label='High Variation')
    plt.axhspan(30, y_limit, color='#E74C3C', alpha=0.2)
    
    plt.xlabel('Syllable Occurrence (with timestamp)', fontsize=12)
    plt.ylabel('Energy Deviation from Mean (%)', fontsize=12)
    plt.title('Syllable Energy Analysis\n(Consistency of Articulatory Effort)', 
              fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    plt.savefig('energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# FILE OPERATIONS FUNCTIONS
# =============================================================================

def create_results_text(syllable_stats, syllable_count, audio_duration, syllables_per_second):
    """Create formatted results text."""
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

    # Update results text to include energy metrics
    results_text += "\n\nENERGY MEASUREMENTS (articulatory effort)"
    results_text += "\n----------------------------------------"
    for syllable, stats in syllable_stats.items():
        results_text += f"""
{syllable.upper()} Syllable Energy:
- Mean energy: {stats['mean_energy']:.3f}
- Energy variation (CV): {stats['energy_cv']:.3f}
- Interpretation:
  * CV < 0.15: Consistent effort
  * CV 0.15-0.30: Moderate variation
  * CV > 0.30: High variation in effort"""

    return results_text

def save_analysis_files(audio_bytes, results_text):
    """Save all analysis files and create zip package."""
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
        zipf.write('evenness_analysis.png')
        zipf.write('distinctness_analysis.png')
        zipf.write('energy_analysis.png')
        zipf.write(audio_path)
        zipf.write('results/analysis_results.txt')
    
    return zip_path

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def get_syllables_per_second(audio_bytes):
    """
    Main analysis function that processes audio and returns syllables per second.
    Uses modular functions for better organization and maintainability.
    """

    # Define hop_length for mel spectrogram display (must match the one used in generation)
    hop_length = 128  # This should match the hop_length used in create_mel_spectrogram

    # Step 1: Load and preprocess audio
    audio_input, sample_rate, processor = load_and_preprocess_audio(audio_bytes)

    # Step 2: Create mel spectrogram
    mel_spec_db = create_mel_spectrogram(audio_input, sample_rate, hop_length)
    
    # Step 3: Transcribe audio and extract syllable offsets
    syllable_offsets, probabilities = transcribe_audio(audio_input, processor)
    
    # Step 4: Calculate basic syllable metrics
    syllable_count, audio_duration, syllables_per_second, syllable_duration = calculate_syllable_metrics(
        syllable_offsets, audio_input, sample_rate
    )

    # Step 5: Create syllables per second plot
    create_syllables_per_second_plot(syllable_offsets, sample_rate)

    # Step 6: Create mel spectrogram plot
    create_mel_spectrogram_plot(mel_spec_db, sample_rate, syllable_offsets, hop_length)

    # Step 7: Calculate detailed syllable statistics
    syllable_stats = calculate_syllable_stats(syllable_offsets, processor, probabilities, sample_rate)

    # Step 8: Create analysis visualizations
    create_evenness_analysis_plot(syllable_stats)
    create_distinctness_analysis_plot(syllable_stats)
    create_energy_analysis_plot(syllable_stats, audio_input, sample_rate)

    # Step 9: Create results text
    results_text = create_results_text(syllable_stats, syllable_count, audio_duration, syllables_per_second)
    
    # Step 10: Print results for verification
    print("\nFinal Results Text:")
    print(results_text)
    
    # Step 11: Save all analysis files
    zip_path = save_analysis_files(audio_bytes, results_text)
    
    # Step 12: Display UI elements
    display_analysis_results(zip_path, syllable_stats)
    
    return syllables_per_second

def display_analysis_results(zip_path, syllable_stats):
    """Display analysis results in the Streamlit UI."""
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
    st.image('evenness_analysis.png')
    st.image('distinctness_analysis.png')
    st.image('energy_analysis.png')
    st.image('mel_spectrogram.png')
    
    # Display detailed metrics
    st.write("### Detailed Analysis")
    for syllable, stats in syllable_stats.items():
        st.write(f"\n**{syllable.upper()} Syllable:**")
        st.write(f"- Count: {stats['count']}")
        st.write(f"- Mean interval: {stats['mean_interval']:.3f} seconds")
        st.write(f"- Coefficient of variation: {stats['cv']:.3f}")
        st.write(f"- Mean articulation confidence: {stats['mean_confidence']:.3f}")

# =============================================================================
# UI FUNCTIONS
# =============================================================================

def process_uploaded_file(uploaded_file):
    """Process an uploaded audio file."""
    with st.spinner("Processing the uploaded audio file..."):
        # Read entire file into bytes
        audio_bytes = uploaded_file.read()
        result = get_syllables_per_second(audio_bytes)
        st.write(f"**Syllables per second (uploaded):** {result:.2f}")

def process_recorded_audio(recorded_data):
    """Process recorded audio from microphone."""
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

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# -----------------------------
# SECTION: File Uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    process_uploaded_file(uploaded_file)

# -----------------------------
# SECTION: Audio Recorder
# -----------------------------
st.write("---")
st.subheader("Or record audio from your microphone")

# The st_audiorec component returns base64 encoded wav data
recorded_data = st_audiorec()

if recorded_data is not None:
    process_recorded_audio(recorded_data)
