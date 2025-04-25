import os

# Function to read reference text (phones) from the corpus folder
def read_reference_text(corpus_folder):
    reference_file = os.path.join(corpus_folder, "reference_phones.txt")  # Ensure the file name is correct
    reference_phones = []
    try:
        with open(reference_file, "r", encoding="utf-8") as f:
            for line in f:
                reference_phones.append(line.strip())  # Read each line and add it to the list
    except FileNotFoundError:
        print(f"Reference file '{reference_file}' not found.")
    return reference_phones

# Function to read MFA output phonemes from the TextGrid file
def read_mfa_phonemes(output_folder):
    output_file = os.path.join(output_folder, "sample.TextGrid")  # Correct the file name to 'sample.TextGrid'
    phonemes = []
    
    try:
        with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()  # Read the entire content of the TextGrid file
            
            # Find intervals with phonemes (assuming phonemes are stored in intervals)
            phoneme_start = content.find('intervals')
            if phoneme_start == -1:
                print("No phoneme intervals found in the TextGrid file.")
                return phonemes

            # Extract the portion containing phonemes
            phoneme_data = content[phoneme_start:]
            
            # Parse phonemes from the extracted portion (this step depends on your TextGrid structure)
            phoneme_lines = phoneme_data.splitlines()
            for line in phoneme_lines:
                if 'phoneme' in line.lower():  # Assuming the word "phoneme" marks phoneme data
                    phonemes.append(line.strip())

    except FileNotFoundError:
        print(f"MFA output file '{output_file}' not found.")
    return phonemes

# Main section of the script
if __name__ == "__main__":
    corpus_folder = "D:/IW-Sys/Whisper/corpus"  # Path to the corpus folder
    output_folder = "D:/IW-Sys/Whisper/output"  # Path to the output folder

    # Read reference and predicted phonemes
    reference_phones = read_reference_text(corpus_folder)  # Read reference from corpus folder
    predicted_phones = read_mfa_phonemes(output_folder)  # Read predicted phonemes from MFA output

    # Print out the phonemes for debugging
    print("Reference Phones: ", reference_phones)
    print("Predicted Phones: ", predicted_phones)
