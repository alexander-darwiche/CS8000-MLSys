import os
import csv
from jiwer import wer

# Configuration
CSV_PATH = "quant_benchmark_results.csv"
GROUND_TRUTH_ROOT = "transcripts"
PREDICTIONS_ROOT = "transcripts_quant"
UPDATED_CSV = "quant_benchmark_results_updated.csv"

def normalize_text(text):
    return text.strip().lower()

def calculate_missing_wer():
    # Read existing CSV data
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Process each row
    for row in rows:
        if row['wer']:  # Skip already calculated WER
            continue
            
        base_name = os.path.splitext(row['file'])[0]
        model = row['model']
        quant_mode = row['quant_mode']
        
        # Path setup
        gt_dir = os.path.join(GROUND_TRUTH_ROOT, base_name)
        pred_dir = os.path.join(PREDICTIONS_ROOT, base_name)
        gt_path = os.path.join(gt_dir, f"{base_name}_REAL.txt")
        pred_path = os.path.join(pred_dir, f"{base_name}_{model}_{quant_mode}.txt")

        # Calculate WER if files exist
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    reference = normalize_text(f.read())
                
                with open(pred_path, 'r', encoding='utf-8') as f:
                    hypothesis = normalize_text(f.read())
                
                row['wer'] = f"{wer(reference, hypothesis):.4f}"
            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                row['wer'] = "ERROR"
        else:
            print(f"Missing files for {base_name}")
            row['wer'] = "MISSING"

    # Write updated CSV
    with open(UPDATED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    calculate_missing_wer()
    print(f"Updated CSV saved to {UPDATED_CSV}")