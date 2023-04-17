import soundfile as sf
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator

def analyze_audio(audio_file):
    audio_data, sample_rate = sf.read(audio_file)
    audio_duration = len(audio_data) / sample_rate

    # Phonemize the audio file
    transcription = phonemize(
        audio_file,
        backend='espeak',
        language='en-us',
        strip=True,
        separator=Separator(phone=' ', word='|', syllable='-'),
        with_start_and_end_times=True)

    # Count syllables
    syllable_count = sum(1 for item in transcription if item['char'] != ' ')

    # Calculate syllables per second
    syllables_per_second = syllable_count / audio_duration

    # Count correct sequences of "patɛkɛ"
    correct_sequence = "patɛkɛ"
    correct_sequences_count = ''.join(item['char'] for item in transcription).lower().count(correct_sequence)

    return syllables_per_second, transcription, correct_sequences_count

# Example usage:
audio_file = "/home/bmoell/ai-speech-pathology/pataka-test/pataka_16k.wav"
syllables_per_second, transcription, correct_sequences_count = analyze_audio(audio_file)

print("Syllables per second:", syllables_per_second)
print("Transcription:", transcription)
print("Number of correct 'patɛkɛ' sequences:", correct_sequences_count)