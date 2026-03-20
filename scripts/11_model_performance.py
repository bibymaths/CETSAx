import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent/"results"

def analyze_results():
    # 1. Load the specific files
    truth_file = Path(base_dir, "nadph_seq_supervised.csv")
    pred_file = Path(base_dir, "predictions_nadph_seq.csv")

    print(f"Loading {truth_file} and {pred_file}...")
    try:
        truth = pd.read_csv(truth_file)
        preds = pd.read_csv(pred_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Merge on 'id' to sync rows
    # suffix _true for ground truth, _pred for predictions
    merged = pd.merge(truth, preds, on="id", suffixes=('_true', '_pred'))
    print(f"Matched {len(merged)} proteins for analysis.\n")

    # 3. Define the specific columns based on your headers
    y_true = merged['label_cls']  # 0, 1 from supervised file
    y_pred = merged['pred_class_idx']  # 0, 1  from predictions

    # 4. Calculate & Print Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"=== Overall Accuracy: {acc:.2%} ===\n")

    # Target names map to 0, 1 (Weak, Strong)
    print("--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Weak', 'Strong']))

    print("--- Confusion Matrix (Row=Actual, Col=Predicted) ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\n" + "=" * 40 + "\n")

    # 5. DIAGNOSTICS: Find the Missed "Strong" Hits
    # We want to know which Strong binders (2) were predicted as Weak (0)
    missed_hits = merged[(merged['label_cls'] == 1) & (merged['pred_class_idx'] == 0)]

    if not missed_hits.empty:
        print(f"WARNING: The model completely missed {len(missed_hits)} Strong hits (predicted as Weak).")
        print("Top 5 Missed IDs:")
        print(missed_hits['id'].head(5).tolist())

        # Save them for inspection
        missed_hits.to_csv(f"{base_dir}/missed_strong_hits.csv", index=False)
        print("Saved full list to 'missed_strong_hits.csv'")
    else:
        print("Great news: No Strong hits were completely missed!")


if __name__ == "__main__":
    analyze_results()