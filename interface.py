import tkinter as tk
import librosa
import pygame

# Initialize pygame mixer
pygame.mixer.init()

def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def on_play_button_click():
    play_audio("pataka_16k.wav")
    root.after(100, update_highlight)

def update_highlight():
    if pygame.mixer.music.get_busy():
        current_time_ms = pygame.mixer.music.get_pos()
        current_char = None

        for idx, char_info in enumerate(char_timestamps):
            if char_info['start_offset'] <= current_time_ms <= char_info['end_offset']:
                current_char = idx
                break

        if current_char is not None:
            text_widget.tag_remove("highlight", "1.0", tk.END)
            text_widget.tag_add("highlight", f"1.{current_char}", f"1.{current_char + 1}")
            text_widget.see(f"1.{current_char}")

        root.after(100, update_highlight)

char_timestamps = [[{'char': 'p', 'start_offset': 64, 'end_offset': 65}, {'char': 'a', 'start_offset': 66, 'end_offset': 67}, {'char': 't', 'start_offset': 69, 'end_offset': 70}, {'char': 'ɛ', 'start_offset': 71, 'end_offset': 72}, {'char': 'k', 'start_offset': 74, 'end_offset': 75}, {'char': 'ɛ', 'start_offset': 76, 'end_offset': 77}, {'char': 'p', 'start_offset': 79, 'end_offset': 80}, {'char': 'a', 'start_offset': 80, 'end_offset': 81}, {'char': 't', 'start_offset': 83, 'end_offset': 85}, {'char': 'ɛ', 'start_offset': 85, 'end_offset': 86}, {'char': 'k', 'start_offset': 88, 'end_offset': 89}, {'char': 'ɛ', 'start_offset': 90, 'end_offset': 91}, {'char': 'p', 'start_offset': 93, 'end_offset': 94}, {'char': 'ɑ', 'start_offset': 95, 'end_offset': 96}, {'char': 't', 'start_offset': 99, 'end_offset': 100}, {'char': 'ɛ', 'start_offset': 100, 'end_offset': 101}, {'char': 'k', 'start_offset': 103, 'end_offset': 104}, {'char': 'ɛ', 'start_offset': 105, 'end_offset': 106}, {'char': 'p', 'start_offset': 108, 'end_offset': 109}, {'char': 'ɑ', 'start_offset': 110, 'end_offset': 111}, {'char': 't', 'start_offset': 114, 'end_offset': 115}, {'char': 'ɛ', 'start_offset': 115, 'end_offset': 116}, {'char': 'k', 'start_offset': 118, 'end_offset': 119}, {'char': 'ɛ', 'start_offset': 120, 'end_offset': 121}, {'char': 'p', 'start_offset': 123, 'end_offset': 124}, {'char': 'ɑ', 'start_offset': 125, 'end_offset': 126}, {'char': 't', 'start_offset': 129, 'end_offset': 130}, {'char': 'k', 'start_offset': 133, 'end_offset': 134}, {'char': 'p', 'start_offset': 138, 'end_offset': 139}, {'char': 't', 'start_offset': 143, 'end_offset': 144}, {'char': 'k', 'start_offset': 149, 'end_offset': 150}, {'char': 'ɛ', 'start_offset': 150, 'end_offset': 151}, {'char': 'p', 'start_offset': 154, 'end_offset': 155}, {'char': 't', 'start_offset': 159, 'end_offset': 160}, {'char': 'ɛ', 'start_offset': 161, 'end_offset': 162}, {'char': 'k', 'start_offset': 164, 'end_offset': 165}, {'char': 'ɛ', 'start_offset': 166, 'end_offset': 167}, {'char': 'p', 'start_offset': 169, 'end_offset': 170}, {'char': 'ɑ', 'start_offset': 171, 'end_offset': 172}, {'char': 't', 'start_offset': 174, 'end_offset': 175}, {'char': 'ɛ', 'start_offset': 176, 'end_offset': 177}, {'char': 'k', 'start_offset': 179, 'end_offset': 180}, {'char': 'ɛ', 'start_offset': 181, 'end_offset': 182}, {'char': 'p', 'start_offset': 184, 'end_offset': 185}, {'char': 'ɑ', 'start_offset': 186, 'end_offset': 187}, {'char': 't', 'start_offset': 189, 'end_offset': 190}, {'char': 'ɛ', 'start_offset': 191, 'end_offset': 192}, {'char': 'k', 'start_offset': 194, 'end_offset': 195}, {'char': 'ɛ', 'start_offset': 196, 'end_offset': 197}, {'char': 'p', 'start_offset': 200, 'end_offset': 201}, {'char': 'a', 'start_offset': 201, 'end_offset': 202}, {'char': 't', 'start_offset': 205, 'end_offset': 206}, {'char': 'k', 'start_offset': 210, 'end_offset': 211}, {'char': 'ɛ', 'start_offset': 212, 'end_offset': 213}, {'char': 'p', 'start_offset': 215, 'end_offset': 216}, {'char': 'a', 'start_offset': 217, 'end_offset': 218}, {'char': 't', 'start_offset': 220, 'end_offset': 221}, {'char': 'ɛ', 'start_offset': 222, 'end_offset': 223}, {'char': 'k', 'start_offset': 225, 'end_offset': 226}, {'char': 'ɛ', 'start_offset': 227, 'end_offset': 228}, {'char': 'p', 'start_offset': 230, 'end_offset': 231}, {'char': 'a', 'start_offset': 232, 'end_offset': 233}, {'char': 't', 'start_offset': 236, 'end_offset': 237}, {'char': 'ɛ', 'start_offset': 238, 'end_offset': 239}, {'char': 'k', 'start_offset': 242, 'end_offset': 243}, {'char': 'ɛ', 'start_offset': 244, 'end_offset': 245}, {'char': 'p', 'start_offset': 247, 'end_offset': 248}, {'char': 'a', 'start_offset': 249, 'end_offset': 250}, {'char': 't', 'start_offset': 253, 'end_offset': 254}, {'char': 'a', 'start_offset': 255, 'end_offset': 256}, {'char': 'k', 'start_offset': 258, 'end_offset': 259}]]

transcription = ''.join([char_info['char'] for char_info in char_timestamps[0]])

# Create tkinter GUI
root = tk.Tk()
root.title("Phoneme Highlighter")

text_widget = tk.Text(root, wrap=tk.WORD)
text_widget.pack(expand=True, fill=tk.BOTH)
text_widget.insert(tk.END, transcription)

text_widget.tag_configure("highlight", background="yellow")

play_button = tk.Button(root, text="Play", command=on_play_button_click)
play_button.pack()

root.mainloop()



