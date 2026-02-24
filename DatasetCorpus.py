import os
import csv
import argparse
from collections import defaultdict

def extract_speaker(fname: str) -> str:
    name = os.path.splitext(os.path.basename(fname))[0]
    tag  = name.split("_")[-1].upper()
    if tag.startswith("F"):
        return "F"
    if tag.startswith("M"):
        return "M"

    parts = fname.replace("\\", "/").split("/")
    for part in reversed(parts[:-1]):
        upper = part.upper()
        if upper.startswith("F"):
            return "F"
        if upper.startswith("M"):
            return "M"

    return "U"

def main(root: str, csv_out: str = "word_counts.csv"):
    word_counts = defaultdict(lambda: {"total": 0, "M": 0, "F": 0, "U": 0})
    total_dirs = 0
    total_files = 0
    total_txt_processed = 0
    for dirpath, _, files in os.walk(root):
        total_dirs += 1
        print(f"[INFO] Entering directory: {dirpath} (total dirs processed: {total_dirs})")
        files_set = set(f.lower() for f in files)  # Case-insensitive set
        txt_files_in_dir = 0
        for f in files:
            total_files += 1
            if f.lower().endswith(".txt"):
                txt_files_in_dir += 1
                base, _ = os.path.splitext(f)
                wav_file = base + ".wav"
                if wav_file.lower() in files_set:
                    txt_path = os.path.join(dirpath, f)
                    print(f"  [INFO] Processing TXT file: {txt_path}")
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        content = tf.read().strip()
                        words = content.split()
                    speaker = extract_speaker(base)
                    print(f"    [INFO] Speaker detected: {speaker}, Words found: {len(words)}")
                    for w in words:
                        w_lower = w.lower()
                        word_counts[w_lower]["total"] += 1
                        if speaker in ("M", "F", "U"):
                            word_counts[w_lower][speaker] += 1
                        else:
                            word_counts[w_lower]["U"] += 1
                    total_txt_processed += 1
        print(f"[INFO] Exiting directory: {dirpath} (TXT files processed in dir: {txt_files_in_dir})")
    print(f"[SUMMARY] Total directories traversed: {total_dirs}")
    print(f"[SUMMARY] Total files encountered: {total_files}")
    print(f"[SUMMARY] Total TXT files processed: {total_txt_processed}")
    
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "total_count", "M_count", "F_count"])
        for word in sorted(word_counts):
            cnt = word_counts[word]
            writer.writerow([word, cnt["total"], cnt["M"], cnt["F"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count words in transcripts with speaker subcounts")
    parser.add_argument("root", help="Root directory to traverse")
    parser.add_argument("--out", default="word_counts.csv", help="Output CSV file")
    args = parser.parse_args()
    main(args.root, args.out)
    print("Script executed successfully. Output saved to", args.out)