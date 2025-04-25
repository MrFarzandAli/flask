import os
import shutil
import subprocess
import whisper
import requests
from flask import Flask, request, jsonify
from textgrid import TextGrid

# === Config ===
CORPUS_DIR = "corpus"
DICTIONARY = "english_us_mfa.dict"
ACOUSTIC_MODEL = "english_mfa"
OUTPUT_DIR = "aligned"

app = Flask(__name__)

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")

def check_grammar(text):
    try:
        url = "https://api.languagetool.org/v2/check"
        data = {'text': text, 'language': 'en-US'}
        response = requests.post(url, data=data)
        issues = response.json().get("matches", [])
        grammar_score = max(0, 100 - len(issues) * 5)
        return grammar_score
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error checking grammar: {str(e)}")

def prepare_corpus(audio_file, transcription_text):
    try:
        if os.path.exists(CORPUS_DIR):
            shutil.rmtree(CORPUS_DIR)
        os.makedirs(CORPUS_DIR)
        filename = os.path.basename(audio_file)
        shutil.copy(audio_file, os.path.join(CORPUS_DIR, filename))
        with open(os.path.join(CORPUS_DIR, filename.replace(".wav", ".lab")), "w", encoding="utf-8") as f:
            f.write(transcription_text)
    except Exception as e:
        raise Exception(f"Error preparing corpus: {str(e)}")

def run_mfa_align():
    try:
        subprocess.run([
            "mfa", "align",
            "--clean",
            "--output_format", "long_textgrid",
            CORPUS_DIR,
            DICTIONARY,
            ACOUSTIC_MODEL,
            OUTPUT_DIR
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"MFA alignment error: {str(e)}")
    except FileNotFoundError as e:
        raise Exception(f"MFA executable not found: {str(e)}")

def analyze_alignment(audio_path):
    try:
        filename = os.path.basename(audio_path).replace(".wav", ".TextGrid")
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError("Alignment file not found.")
        
        tg = TextGrid.fromFile(path)
        word_tier, phone_tier = tg[0], tg[1]

        total_words = len([w for w in word_tier if w.mark.strip()])
        slow_words = sum(1 for w in word_tier if w.mark.strip() and (w.maxTime - w.minTime > 1.5))

        total_phones = len([p for p in phone_tier if p.mark.strip() not in ["", "sil", "sp"]])
        long_phones = sum(1 for p in phone_tier if (p.maxTime - p.minTime > 0.2) and p.mark.strip() not in ["", "sil", "sp"])

        fluency = max(0, 100 - (slow_words / total_words * 100)) if total_words else 0
        accuracy = max(0, 100 - (long_phones / total_phones * 100)) if total_phones else 0
        return fluency, accuracy
    except Exception as e:
        raise Exception(f"Error analyzing alignment: {str(e)}")

def phoneme_wise_scores(audio_path):
    try:
        filename = os.path.basename(audio_path).replace(".wav", ".TextGrid")
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError("TextGrid file not found.")

        tg = TextGrid.fromFile(path)
        phone_tier = tg[1]
        scores = []
        for p in phone_tier:
            mark = p.mark.strip()
            duration = p.maxTime - p.minTime
            if mark in ["", "sil", "sp"]:
                continue
            ideal_duration = 0.06
            diff = abs(duration - ideal_duration)
            score = max(0, 1 - diff / 0.1)
            scores.append(score * 100)
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        raise Exception(f"Error calculating phoneme scores: {str(e)}")

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files['audio']
        audio_path = os.path.join("uploads", audio_file.filename)
        os.makedirs("uploads", exist_ok=True)
        audio_file.save(audio_path)

        # Run the processing steps
        text = transcribe_audio(audio_path)
        grammar_score = check_grammar(text)
        prepare_corpus(audio_path, text)
        run_mfa_align()
        fluency, accuracy = analyze_alignment(audio_path)
        pronunciation = phoneme_wise_scores(audio_path)

        return jsonify({
            "fluency": round(fluency, 2),
            "accuracy": round(accuracy, 2),
            "pronunciation": round(pronunciation, 2),
            "grammar": round(grammar_score, 2),
            "transcription": text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
