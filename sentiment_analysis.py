import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

"""
Sentiment Analysis on IMDB Reviews
Using Hugging Face Transformers Pipeline

Models compared:
  - distilbert-base-uncased-finetuned-sst-2-english  (~88%)
  - siebert/sentiment-roberta-large-english           (~94.6%)
"""

import time
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Step 1: Load IMDB Dataset
# ============================================================

csv_path = "./datasets/imdb_top_500.csv"

if not os.path.exists(csv_path):
    print("imdb_top_500.csv not found, generating from datasets library...")
    from datasets import load_dataset
    dataset = load_dataset("imdb", split="test")
    pos = [item for item in dataset if item["label"] == 1][:250]
    neg = [item for item in dataset if item["label"] == 0][:250]
    all_data = pos + neg
    df_gen = pd.DataFrame({
        "text": [item["text"] for item in all_data],
        "label": [item["label"] for item in all_data],
        "rating": [8 if item["label"] == 1 else 4 for item in all_data]
    })
    os.makedirs("./datasets", exist_ok=True)
    df_gen.to_csv(csv_path, index=False)
    print(f"Generated {csv_path} with {len(df_gen)} rows.")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} reviews.")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# ============================================================
# Step 2: Define Models to Compare
# ============================================================

models_to_test = {
    "distilbert-sst2": {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "label_map": {"POSITIVE": 1, "NEGATIVE": 0},
    },
    "roberta-large-siebert": {
        "model": "siebert/sentiment-roberta-large-english",
        "label_map": {"POSITIVE": 1, "NEGATIVE": 0},
    },
}

# ============================================================
# Step 3: Evaluate Each Model
# ============================================================

texts = df["text"].tolist()
true_labels = df["label"].values

MAX_CHAR_LEN = 1500
texts_truncated = [t[:MAX_CHAR_LEN] for t in texts]

results = {}

for name, config in models_to_test.items():
    print(f"\n{'='*60}")
    print(f"Testing model: {name}")
    print(f"  Full name: {config['model']}")
    print(f"{'='*60}")

    try:
        classifier = pipeline(
            task="sentiment-analysis",
            model=config["model"],
            tokenizer=config["model"],
            framework="pt",
            truncation=True,
            max_length=512,
        )

        start_time = time.time()

        batch_size = 16
        all_preds = []

        for i in range(0, len(texts_truncated), batch_size):
            batch = texts_truncated[i:i+batch_size]
            batch_results = classifier(batch)
            batch_preds = [
                config["label_map"].get(r["label"], 0)
                for r in batch_results
            ]
            all_preds.extend(batch_preds)

            done = min(i + batch_size, len(texts_truncated))
            if done % 100 == 0 or done == len(texts_truncated):
                print(f"  Processed {done}/{len(texts_truncated)} reviews...")

        elapsed = time.time() - start_time
        preds = np.array(all_preds)

        acc = accuracy_score(true_labels, preds)
        print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Time: {elapsed:.1f}s")
        print(f"\n  Classification Report:")
        print(classification_report(
            true_labels, preds,
            target_names=["Negative", "Positive"]
        ))

        results[name] = {
            "accuracy": acc,
            "time": elapsed,
            "predictions": preds
        }

    except Exception as e:
        print(f"  Error: {e}")
        results[name] = {"accuracy": 0, "time": 0, "error": str(e)}

# ============================================================
# Step 4: Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

for name, res in results.items():
    if "error" in res:
        print(f"  {name}: ERROR - {res['error']}")
    else:
        status = "PASS" if res["accuracy"] >= 0.95 else "FAIL"
        print(f"  {name}: {res['accuracy']*100:.2f}% accuracy | {res['time']:.1f}s | {status} (>=95%)")

best_name = max(
    [k for k in results if "error" not in results[k]],
    key=lambda k: results[k]["accuracy"],
    default=None
)

if best_name:
    best = results[best_name]
    print(f"\nBest Model: {best_name} with {best['accuracy']*100:.2f}% accuracy")
