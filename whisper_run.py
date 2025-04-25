import whisper
import re

# Helper function to calculate fluency score
def rate_fluency(transcribed_text):
    # Heuristic 1: Sentence completeness (simple example)
    # We'll count full stops as a proxy for sentence completeness
    sentences = re.split(r'\.|\?|\!|\n', transcribed_text)  # Split based on sentence-ending punctuation
    complete_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 3]  # Filter short fragments
    
    # Heuristic 2: Grammar (simple check for basic grammar errors)
    # A more complex model could check grammar more thoroughly, but here we will just look for run-on sentences
    run_on_sentences = len([sentence for sentence in complete_sentences if len(sentence.split()) > 20])  # Lengthy sentences
    
    # Rate fluency based on number of complete sentences and run-ons
    fluency_score = 10 - min(run_on_sentences, 10)  # Limit fluency score to max 10
    
    return fluency_score

# Main code
try:
    print("Loading Whisper model...")
    model = whisper.load_model("small")  # Or "medium" / "large" for better accuracy
    print("Model loaded.")
    
    print("Transcribing audio...")
    result = model.transcribe("sample.wav")  # Replace with the correct path to your audio file

    transcribed_text = result["text"]
    print("\n📜 Transcribed Text:\n", transcribed_text)

    # Calculate fluency score out of 10
    fluency_score = rate_fluency(transcribed_text)
    print(f"\nFluency Score (out of 10): {fluency_score}")

except Exception as e:
    print(f"Error during transcription or rating fluency: {e}")
