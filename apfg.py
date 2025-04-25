import os
import shutil
import subprocess
import whisper
import requests
from textgrid import TextGrid
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Config ===
CORPUS_DIR = "corpus"
DICTIONARY = "english_us_mfa.dict"
ACOUSTIC_MODEL = "english_mfa"
OUTPUT_DIR = "aligned"

# === File Picker ===
def pick_audio_file():
    Tk().withdraw()
    print("📁 Choose your audio file...")
    file_path = askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        print("❌ No file selected.")
        exit()
    return file_path

# === Transcription ===
def transcribe_audio(audio_path):
    print("🎙️ Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    print("📄 Transcribed Text:", text)
    return text

# === Grammar Check ===
def check_grammar(text):
    print("🧠 Checking grammar...")
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': text,
        'language': 'en-US'
    }

    response = requests.post(url, data=data)
    result = response.json()

    grammar_issues = result.get("matches", [])
    num_errors = len(grammar_issues)

    print(f"📝 Grammar Issues Found: {num_errors}")
    for issue in grammar_issues:
        print(f" - ❌ {issue['message']}")
        print(f"   ↪ Suggestion: {', '.join([rep['value'] for rep in issue['replacements']])}")
        print(f"   🧠 Context: \"{issue['context']['text']}\"\n")

    grammar_score = max(0, 100 - num_errors * 5)  # Deduct 5 points per issue
    print(f"✅ Grammar Score: {grammar_score:.2f}/100")
    return grammar_score

# === Corpus Prep ===
def prepare_corpus(audio_file, transcription_text):
    print("📂 Setting up corpus...")
    if os.path.exists(CORPUS_DIR):
        shutil.rmtree(CORPUS_DIR)
    os.makedirs(CORPUS_DIR)

    filename = os.path.basename(audio_file)
    dest_audio = os.path.join(CORPUS_DIR, filename)
    shutil.copy(audio_file, dest_audio)

    lab_path = os.path.join(CORPUS_DIR, filename.replace(".wav", ".lab"))
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)

# === MFA Align ===
def run_mfa_align():
    print("⚙️ Running MFA alignment...")
    command = [
        "mfa", "align",
        "--clean",
        "--output_format", "long_textgrid",
        CORPUS_DIR,
        DICTIONARY,
        ACOUSTIC_MODEL,
        OUTPUT_DIR
    ]
    subprocess.run(command, check=True)

# === Analyze Alignment ===
def analyze_alignment(audio_path):
    print("📊 Analyzing alignment...")
    filename = os.path.basename(audio_path).replace(".wav", ".TextGrid")
    textgrid_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(textgrid_path):
        print("❌ Alignment failed, TextGrid not found.")
        return 0, 0

    tg = TextGrid.fromFile(textgrid_path)
    word_tier = tg[0]
    phone_tier = tg[1]

    total_words = len([w for w in word_tier if w.mark.strip()])
    slow_words = sum(1 for w in word_tier if w.mark.strip() and (w.maxTime - w.minTime > 1.5))

    total_phones = len([p for p in phone_tier if p.mark.strip() not in ["", "sil", "sp"]])
    long_phones = sum(1 for p in phone_tier if (p.maxTime - p.minTime > 0.2) and p.mark.strip() not in ["", "sil", "sp"])

    fluency = max(0, 100 - (slow_words / total_words * 100)) if total_words else 0
    accuracy = max(0, 100 - (long_phones / total_phones * 100)) if total_phones else 0

    print(f"✅ Fluency Score: {fluency:.2f}")
    print(f"✅ Accuracy Score: {accuracy:.2f}")
    return fluency, accuracy

# === Phoneme Scoring ===
def phoneme_wise_scores(audio_path):
    filename = os.path.basename(audio_path).replace(".wav", ".TextGrid")
    textgrid_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(textgrid_path):
        print("❌ TextGrid not found for phoneme scoring.")
        return []

    tg = TextGrid.fromFile(textgrid_path)
    phone_tier = tg[1]
    scores = []

    print("\n🔍 Phoneme-wise Scoring:")
    for p in phone_tier:
        mark = p.mark.strip()
        duration = p.maxTime - p.minTime
        if mark in ["", "sil", "sp"]:
            continue

        ideal_duration = 0.06  # ~60ms is average phoneme time
        diff = abs(duration - ideal_duration)
        score = max(0, 1 - diff / 0.1)  # Tolerate up to 100ms difference
        percent = score * 100
        scores.append(percent)

        print(f"  🔤 {mark:3s} | Duration: {duration:.2f}s | Score: {percent:.1f}/100")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n🎯 Average Phoneme Score: {avg_score:.2f}/100")
    else:
        avg_score = 0
        print("⚠️ No phonemes detected.")

    return scores

# === Main Evaluation Logic ===
def evaluate():
    audio_path = pick_audio_file()
    text = transcribe_audio(audio_path)
    grammar_score = check_grammar(text)  # 👈 Grammar check here
    prepare_corpus(audio_path, text)
    run_mfa_align()
    fluency, accuracy = analyze_alignment(audio_path)
    phoneme_scores = phoneme_wise_scores(audio_path)
    phoneme_score = sum(phoneme_scores) / len(phoneme_scores) if phoneme_scores else 0
    return fluency, accuracy, phoneme_score, grammar_score

# === Entry Point ===
if __name__ == "__main__":
    fluency, accuracy, pronunciation, grammar = evaluate()

    print("\n🎯 Final Evaluation:")
    print(f"   🗣️ Fluency:         {fluency:.2f}/100")
    print(f"   🎯 Accuracy:        {accuracy:.2f}/100")
    print(f"   🔤 Pronunciation:   {pronunciation:.2f}/100")
    print(f"   📚 Grammar:         {grammar:.2f}/100")
