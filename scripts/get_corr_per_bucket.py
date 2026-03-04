import jsonlines
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import argparse as ap

from pathlib import Path
import argparse as ap

import numpy as np
from nlpstats.correlations import correlate, bootstrap

from evaluation.utils import load_predictions, load_gold_data


# Initialize nested defaultdict to avoid manual initialization of scores
from collections import defaultdict
label_distribution = defaultdict(lambda: defaultdict(int))
        

def main():

    parser = ap.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to input folder of containing the gold annotations for get-doc-len-buckets. for get-corr-per-bucket, it is the path to the predictions.")
    parser.add_argument("--get-doc-len-buckets", action="store_true", help="Prints the number of documents per bucket, where one bucket is a range of token lengths.")
    parser.add_argument("--get-corr-per-bucket", action="store_true", help="Prints the correlation per bucket.")
    parser.add_argument("--lang", type=str, help="Language to get the correlation for. Is needed for the correlation per bucket.")
    args = parser.parse_args()

    with open("list_to_drop.txt", "r") as f:
        list_to_drop = f.read().splitlines()
        list_to_drop = [_.strip() for _ in list_to_drop]

    token_buckets_count = defaultdict(int)

    if args.get_doc_len_buckets:
        for lang in ["de", "fr", "it"]:

            
            with jsonlines.open(f"{args.input_folder}/gold_admin_{lang}.jsonl") as reader:
                for item in reader:

                    if item['id'] in list_to_drop and ("DiffAlign" not in args.input_folder or "ModernBERT" not in args.input_folder):
                        continue

                    if len(item['labels_a']) < 200:
                        token_buckets_count["<200"] += 1
                    if len(item['labels_b']) < 200:
                        token_buckets_count["<200"] += 1
                    if len(item['labels_a']) >= 200 and len(item['labels_a']) < 400:
                        token_buckets_count["200-400"] += 1
                    if len(item['labels_b']) >= 200 and len(item['labels_b']) < 400:
                        token_buckets_count["200-400"] += 1
                    if len(item['labels_a']) >= 400 and len(item['labels_a']) < 600:
                        token_buckets_count["400-600"] += 1
                    if len(item['labels_b']) >= 400 and len(item['labels_b']) < 600:
                        token_buckets_count["400-600"] += 1
                    if len(item['labels_a']) >= 600 and len(item['labels_a']) < 800:
                        token_buckets_count["600-800"] += 1
                    if len(item['labels_b']) >= 600 and len(item['labels_b']) < 800:
                        token_buckets_count["600-800"] += 1
                    if len(item['labels_a']) >= 800 and len(item['labels_a']) < 1000:
                        token_buckets_count["800-999"] += 1
                    if len(item['labels_b']) >= 800 and len(item['labels_b']) < 1000:
                        token_buckets_count["800-999"] += 1
                    if len(item['labels_a']) >= 1000 and len(item['labels_a']) < 1200:
                        token_buckets_count["1000-1199"] += 1
                    if len(item['labels_b']) >= 1000 and len(item['labels_b']) < 1200:
                        token_buckets_count["1000-1199"] += 1
                    if len(item['labels_a']) >= 1200 and len(item['labels_a']) < 1400:
                        token_buckets_count["1200-1399"] += 1
                    if len(item['labels_b']) >= 1200 and len(item['labels_b']) < 1400:
                        token_buckets_count["1200-1399"] += 1
                    if len(item['labels_a']) >= 1400 and len(item['labels_a']) < 1600:
                        token_buckets_count["1400-1599"] += 1
                    if len(item['labels_b']) >= 1400 and len(item['labels_b']) < 1600:
                        token_buckets_count["1400-1599"] += 1
                    if len(item['labels_a']) >= 1600 and len(item['labels_a']) < 1800:
                        token_buckets_count["1600-1799"] += 1
                    if len(item['labels_b']) >= 1600 and len(item['labels_b']) < 1800:
                        token_buckets_count["1600-1799"] += 1
                    if len(item['labels_a']) >= 1800 and len(item['labels_a']) < 2000:
                        token_buckets_count["1800-1999"] += 1
                    if len(item['labels_b']) >= 1800 and len(item['labels_b']) < 2000:
                        token_buckets_count["1800-1999"] += 1
                    if len(item['labels_a']) >= 2000:
                        token_buckets_count[">2000"] += 1
                    if len(item['labels_b']) >= 2000:
                        token_buckets_count[">2000"] += 1

        print("\nToken length distribution:")
        print("-" * 40)
        print(f"{'Bucket':15} | {'Count':>10}")
        print("-" * 40)
        for bucket in sorted(token_buckets_count.keys(), key=lambda x: float(x.replace(">", "").split("-")[0].replace("<", ""))):
            print(f"{bucket:15} | {token_buckets_count[bucket]:>10}")
        print("-" * 40)
    
    if args.get_corr_per_bucket:
        # Define fixed windows up to 2000 tokens
        window_size = 200
        num_windows = 11  # 0-200, 200-400, ..., 1800-2000, >2000

        # Process each window across all samples
        correlations = []
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size
            
            window_pred_labels = []
            window_gold_labels = []

            for lang in ["de", "fr", "it"]:
                #predictions_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions' / f'llama-405b-0-236-admin-{lang}.jsonl' # change path according to the model predictions you want to analyze
                #predictions_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'encoder_predictions' / f'DiffAlign(model=Qwen_Qwen3-Embedding-8B, layer=-1_admin_{lang}.jsonl.jsonl' # change path according to the model predictions you want to analyze
                # /local/scratch/mwastl/benchmarking-rsd-x/data/evaluation/encoder_predictions/DiffAlign(model=Alibaba-NLP_gte-multilingual-base, layer=-1_admin_de.jsonl.jsonl
                predictions_path = args.input_folder + f"{lang}.jsonl.jsonl"
                predictions = load_predictions(predictions_path)
                """print(len(predictions))"""

                if "ModernBERT" not in str(predictions_path) and "DiffAlign" not in str(predictions_path):
                    print("Dropping predictions")
                    predictions = [prediction for prediction in predictions if prediction.item_id not in list_to_drop]
                gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'full' /f'gold_admin_{lang}.jsonl' if "full" in args.input_folder else Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'dev' /f'gold_admin_{lang}.jsonl'
                gold_samples = load_gold_data(gold_path)

                """print(len(gold_samples))
                print(len(predictions))"""
                assert len(predictions) == len(gold_samples)

            
                
                counter = 0
                counter_trunc = 0
                for prediction, gold_sample in zip(predictions, gold_samples):
                    pred_labels_a = prediction.get_difference_sample().labels_a
                    gold_labels_a = gold_sample.labels_a

                    # filter out predictions that are too short
                    if len(pred_labels_a) >= start_idx:
                        window_range_a = pred_labels_a[start_idx:end_idx]
                        gold_range_a = gold_labels_a[start_idx:end_idx]
                    else:
                        continue

                    # Pad window ranges with 0 to match lengths
                    if len(window_range_a) < len(gold_range_a):
                        window_range_a = window_range_a + (0,) * (len(gold_range_a) - len(window_range_a))
                        counter += 1
                    elif len(window_range_a) > len(gold_range_a):
                        gold_range_a = gold_range_a + (0,) * (len(window_range_a) - len(gold_range_a))
                        counter_trunc += 1

                    if window_range_a and gold_range_a:  # Only add if we have labels in this range
                        window_pred_labels.extend(window_range_a)
                        window_gold_labels.extend(gold_range_a)

                    # Process labels_b if they exist
                    gold_labels_b = gold_sample.labels_b
                    if not all(label == -1 for label in gold_labels_b):
                        pred_labels_b = prediction.get_difference_sample().labels_b
                        
                        window_range_b = pred_labels_b[start_idx:end_idx]
                        gold_range_b = gold_labels_b[start_idx:end_idx]

                        # Pad window ranges with 0 to match lengths
                        if len(window_range_b) < len(gold_range_b):
                            window_range_b = window_range_b + (0,) * (len(gold_range_b) - len(window_range_b))
                            counter += 1
                        elif len(window_range_b) > len(gold_range_b):
                            gold_range_b = gold_range_b + (0,) * (len(window_range_b) - len(gold_range_b))
                            counter_trunc += 1

                        if window_range_b and gold_range_b:  # Only add if we have labels in this range
                            window_pred_labels.extend(window_range_b)
                            window_gold_labels.extend(gold_range_b)
            
            """print(f"Number of predictions that were padded: {counter}")
            print(f"Number of predictions that were truncated: {counter_trunc}")"""

            if not window_pred_labels or not window_gold_labels:
                correlations.append("NA")
                continue

            # Filter out -1 labels
            filtered_pairs = [(pred, gold) for pred, gold in zip(window_pred_labels, window_gold_labels) 
                            if gold != -1]
            if not filtered_pairs:
                correlations.append("NA") 
                continue
                
            filtered_pred_labels, filtered_gold_labels = zip(*filtered_pairs)

            pred_array = np.expand_dims(np.array(filtered_pred_labels), 0)
            gold_array = np.expand_dims(np.array(filtered_gold_labels), 0)

            assert len(pred_array) > 0 and len(gold_array) > 0, f"No labels in window {start_idx}-{end_idx}"

            if np.unique(pred_array).size == 1:
                print(len(pred_array))
                pred_array[0][0] = 0.5
                print(f"In window {start_idx}-{end_idx} all predicted labels are the same\n. \n Introducing a miniscule variation")
                print(f"pred_array: {np.unique(pred_array)}")

            if np.unique(gold_array).size == 1:
                print(len(gold_array))
                gold_array[0][0] = 0.5
                print(f"In window {start_idx}-{end_idx} all gold labels are the same\n. \n Introducing a miniscule variation")
                print(f"gold_array: {np.unique(gold_array)}")
            # assert whether there is nan in the arrays with an assertion
            assert not np.isnan(pred_array).any(), f"There is nan in the pred array in window {start_idx}-{end_idx}"
            assert not np.isnan(gold_array).any(), f"There is nan in the gold array in window {start_idx}-{end_idx}"


            correlation = correlate(
                pred_array,
                gold_array,
                level="global",
                coefficient="spearman",
            )
            correlations.append(f"{correlation*100:.1f}")

        # Print correlations in a column format
        print(f"\nCorrelations per window for {str(predictions_path).split('/')[-1]}:")
        print("Window\tCorrelation")
        print("-" * 25)
        windows = ["0-200", "200-400", "400-600", "600-800", "800-1000", 
                  "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", ">2000"]
        for window, corr in zip(windows, correlations):
            print(f"{window}\t{corr}")
                        







if __name__ == "__main__":
    main()