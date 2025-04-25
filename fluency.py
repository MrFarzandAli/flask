import whisper
import time
import numpy as np

# Load the Whisper model
model = whisper.load_model("base")

# Path to your audio file and output text file
audio_path = "D:/IW-Sys/Whisper/Sample2.wav"
output_path = "D:/IW-Sys/Whisper/save.txt"

# Transcribe the audio file
start_time = time.time()
result = model.transcribe(audio_path)
end_time = time.time()

# Fluency Score: Calculate the duration of the audio and speech rate
audio_duration = end_time - start_time  # in seconds
word_count = len(result['text'].split())
speech_rate = word_count / audio_duration  # words per second (WPS)

# Pronunciation Score: Check phoneme-based similarity (simulated with simple error checking for now)
pronunciation_score = 0.0
for word in result['text'].split():
    if word in ["the", "a", "is", "are"]:  # Simple error checking example (improve with phoneme models)
        pronunciation_score += 1  # Correct pronunciation count

# Normalize pronunciation score
max_pronunciation_score = word_count  # Maximum score if every word was pronounced correctly
pronunciation_score = (pronunciation_score / max_pronunciation_score) * 100  # percentage

# Output the results to the text file
with open(output_path, "w") as file:
    file.write(f"Transcription: {result['text']}\n")
    file.write(f"Fluency Score: {speech_rate:.2f} words per second\n")
    file.write(f"Pronunciation Score: {pronunciation_score:.2f}%\n")

print(f"Results saved to {output_path}")
