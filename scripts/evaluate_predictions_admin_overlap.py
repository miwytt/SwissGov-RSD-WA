from pathlib import Path
import argparse as ap
import os

import numpy as np
from nlpstats.correlations import correlate, bootstrap

from evaluation.utils import load_predictions, load_gold_data


def compute_correlations(pred_labels, gold_labels, show_std: bool):
    """Compute Spearman and Kendall correlations for given prediction and gold labels."""
    # Filter out labels where gold is -1
    filtered_pred_labels = [pred for pred, gold in zip(pred_labels, gold_labels) if gold != -1]
    filtered_gold_labels = [gold for pred, gold in zip(pred_labels, gold_labels) if gold != -1]

    assert len(filtered_pred_labels) == len(filtered_gold_labels)

    pred_array = np.expand_dims(np.array(filtered_pred_labels), 0)
    gold_array = np.expand_dims(np.array(filtered_gold_labels), 0)

    # Compute Spearman correlation
    spearman_correlation = correlate(
        pred_array,
        gold_array,
        level="global",
        coefficient="spearman",
    )
    spearman_bootstrap_result = bootstrap(
        pred_array,
        gold_array,
        level="global",
        coefficient="spearman",
        resampling_method="inputs",
        n_resamples=1,
    )
    spearman_max_interval = max(spearman_bootstrap_result.upper - spearman_correlation, spearman_correlation - spearman_bootstrap_result.lower)
    
    # Compute Kendall's tau-b
    kendall_correlation = correlate(
        pred_array,
        gold_array,
        level="global",
        coefficient="kendall",
    )
    kendall_bootstrap_result = bootstrap(
        pred_array,
        gold_array,
        level="global",
        coefficient="kendall",
        resampling_method="inputs",
        n_resamples=1,
    )
    kendall_max_interval = max(kendall_bootstrap_result.upper - kendall_correlation, kendall_correlation - kendall_bootstrap_result.lower)
    
    return (spearman_correlation, spearman_max_interval), (kendall_correlation, kendall_max_interval)


def main(short: bool, show_std: bool, split_annotators: bool = False):
    # Define all languages to process
    languages = ["de", "fr", "it"]
    split = None  # Fixed split name for overlap evaluation
    
    if split_annotators:
        # Process each language and annotator separately
        all_results = {}  # {lang: {annotator: (spearman, kendall)}}
        
        for lang in languages:
            print(f"=== Processing {lang.upper()} ===")
            
            predictions_path = Path(__file__).parent.parent / 'data' / 'annotations' / 'gold_labels_overlap' / 'annotations2' / f'gold_admin_{lang}{"_short" if short else ""}.jsonl'
            predictions = load_predictions(predictions_path)
            print(len(predictions))

            gold_path = Path(__file__).parent.parent / 'data' / 'annotations' / 'gold_labels_overlap' / 'annotations1' / f'gold_admin_{lang}{"_short" if short else ""}.jsonl'
            gold_samples = load_gold_data(gold_path)

            print(len(gold_samples))
            print(len(predictions))
            assert len(predictions) == len(gold_samples)

            # Group data by annotator
            annotator_data = {}  # {annotator_tag: (predictions, gold_samples)}
            
            for prediction, gold_sample in zip(predictions, gold_samples):
                annotator_tag = gold_sample.annotator_tag
                if annotator_tag is None:
                    annotator_tag = "unknown"
                else:
                    annotator_tag = str(annotator_tag)  # Convert to string for consistent handling
                
                if annotator_tag not in annotator_data:
                    annotator_data[annotator_tag] = ([], [])
                
                annotator_data[annotator_tag][0].append(prediction)
                annotator_data[annotator_tag][1].append(gold_sample)
            
            # Process each annotator separately
            lang_results = {}
            for annotator_tag, (annotator_predictions, annotator_gold_samples) in annotator_data.items():
                print(f"  Processing annotator {annotator_tag} ({len(annotator_predictions)} samples)")
                
                pred_labels = []
                gold_labels = []
                counter = 0
                counter_b = 0
                
                for prediction, gold_sample in zip(annotator_predictions, annotator_gold_samples):
                    pred_labels_a = prediction.get_difference_sample().labels_a
                    gold_labels_a = gold_sample.labels_a
                    
                    if len(pred_labels_a) < len(gold_labels_a):
                        pred_labels_a = pred_labels_a + (0.0,) * (len(gold_labels_a) - len(pred_labels_a))
                        counter += 1
                    elif len(pred_labels_a) > len(gold_labels_a):
                        pred_labels_a = pred_labels_a[:len(gold_labels_a)]
                        counter += 1
                    
                    assert len(pred_labels_a) == len(gold_labels_a), f"{len(pred_labels_a)} != {len(gold_labels_a)}"
                    pred_labels.extend(pred_labels_a)
                    gold_labels.extend(gold_labels_a)

                    gold_labels_b = gold_sample.labels_b
                    if not all(label == -1 for label in gold_labels_b):
                        pred_labels_b = prediction.get_difference_sample().labels_b
                        
                        if len(pred_labels_b) < len(gold_labels_b):
                            counter_b += 1
                            pred_labels_b = pred_labels_b + (0.0,) * (len(gold_labels_b) - len(pred_labels_b))
                        elif len(pred_labels_b) > len(gold_labels_b):
                            counter_b += 1
                            pred_labels_b = pred_labels_b[:len(gold_labels_b)]

                        assert len(pred_labels_b) == len(gold_labels_b), f"{len(pred_labels_b)} != {len(gold_labels_b)}"
                        pred_labels.extend(pred_labels_b)
                        gold_labels.extend(gold_labels_b)
                
                if counter > 0:
                    print(f"    Number of samples that did not have the label length a: {counter}")
                if counter_b > 0:
                    print(f"    Number of samples that did not have the label length b: {counter_b}")
                
                # Compute correlations for this annotator
                spearman_result, kendall_result = compute_correlations(pred_labels, gold_labels, show_std)
                lang_results[annotator_tag] = (spearman_result, kendall_result)
            
            all_results[lang] = lang_results
        
        # Print results by annotator
        print(f"\n TEST RESULTS BY ANNOTATOR===")
        
        # Get all unique annotators across languages
        all_annotators = set()
        for lang_results in all_results.values():
            all_annotators.update(lang_results.keys())
        
        # Sort annotators with numeric values first, then string values
        def sort_key(annotator):
            try:
                # Try to convert to int for numeric sorting
                return (0, int(annotator))
            except ValueError:
                # If not numeric, sort as string
                return (1, annotator)
        
        all_annotators = sorted(all_annotators, key=sort_key)
        
        # Print Spearman results
        print("\nSpearman:")
        print("\t".join([f"{lang}_{annotator}" for lang in languages for annotator in all_annotators]))
        # Create a single row with all values
        row_values = []
        for lang in languages:
            for annotator in all_annotators:
                if annotator in all_results[lang]:
                    corr, std = all_results[lang][annotator][0]
                    if show_std:
                        row_values.append(f"{corr:.3f}±{std:.3f}")
                    else:
                        row_values.append(f"{corr:.3f}")
                else:
                    row_values.append("N/A")
        print("\t".join(row_values))
        
        # Print Kendall results
        print("\nKendall's Tau-b:")
        print("\t".join([f"{lang}_{annotator}" for lang in languages for annotator in all_annotators]))
        # Create a single row with all values
        row_values = []
        for lang in languages:
            for annotator in all_annotators:
                if annotator in all_results[lang]:
                    corr, std = all_results[lang][annotator][1]
                    if show_std:
                        row_values.append(f"{corr:.3f}±{std:.3f}")
                    else:
                        row_values.append(f"{corr:.3f}")
                else:
                    row_values.append("N/A")
        print("\t".join(row_values))
    
    else:
        # Original behavior - process all data together
        spearman_results = {}
        kendall_results = {}
        
        for lang in languages:
            print(f"=== Processing {lang.upper()} ===")
            
            predictions_path = Path(__file__).parent.parent / 'data' / 'annotations' / 'gold_labels_overlap' / 'annotations2' / f'gold_admin_{lang}{"_short" if short else ""}.jsonl'
            predictions = load_predictions(predictions_path)
            print(len(predictions))

            gold_path = Path(__file__).parent.parent / 'data' / 'annotations' / 'gold_labels_overlap' / 'annotations1' / f'gold_admin_{lang}{"_short" if short else ""}.jsonl'
            gold_samples = load_gold_data(gold_path)

            print(len(gold_samples))
            print(len(predictions))
            assert len(predictions) == len(gold_samples)

            pred_labels = []
            gold_labels = []
            counter = 0
            counter_b = 0
            unprocessable_counter = 0
            
            for prediction, gold_sample in zip(predictions, gold_samples):
                pred_labels_a = prediction.get_difference_sample().labels_a
                gold_labels_a = gold_sample.labels_a
                
                if len(pred_labels_a) < len(gold_labels_a):
                    pred_labels_a = pred_labels_a + (0.0,) * (len(gold_labels_a) - len(pred_labels_a))
                    counter += 1
                elif len(pred_labels_a) > len(gold_labels_a):
                    pred_labels_a = pred_labels_a[:len(gold_labels_a)]
                    counter += 1
                    print(f"prediction.sentence1: {prediction.sentence1}")
                    print(f"gold_sample.tokens_a: {' '.join(gold_sample.tokens_a)}")
                    print(f"pred_labels_a: {pred_labels_a}")
                    print(f"gold_labels_a: {gold_labels_a}")
                    print()
                
                assert len(pred_labels_a) == len(gold_labels_a), f"{len(pred_labels_a)} != {len(gold_labels_a)}"
                pred_labels.extend(pred_labels_a)
                gold_labels.extend(gold_labels_a)

                gold_labels_b = gold_sample.labels_b
                if not all(label == -1 for label in gold_labels_b):
                    pred_labels_b = prediction.get_difference_sample().labels_b
                    
                    if len(pred_labels_b) < len(gold_labels_b):
                        counter_b += 1
                        pred_labels_b = pred_labels_b + (0.0,) * (len(gold_labels_b) - len(pred_labels_b))
                    elif len(pred_labels_b) > len(gold_labels_b):
                        counter_b += 1
                        pred_labels_b = pred_labels_b[:len(gold_labels_b)]

                    assert len(pred_labels_b) == len(gold_labels_b), f"{len(pred_labels_b)} != {len(gold_labels_b)}"
                    pred_labels.extend(pred_labels_b)
                    gold_labels.extend(gold_labels_b)
            
            if counter > 0:
                print(f"Number of samples that did not have the label length a: {counter}")
            if counter_b > 0:
                print(f"Number of samples that did not have the label length b: {counter_b}")
            if unprocessable_counter > 0:
                print(f"Number of unprocessable samples: {unprocessable_counter}")
            
            # Compute correlations
            spearman_result, kendall_result = compute_correlations(pred_labels, gold_labels, show_std)
            spearman_results[lang] = spearman_result
            kendall_results[lang] = kendall_result
        
        # Print results in table format for easy copy-paste to Google Sheets
        print(f"\n===OVERLAP TEST RESULTS===")
        # Tab-separated values for easy copy-paste
        print("Spearman:")
        row_values = []
        for lang in languages:
            corr, std = spearman_results[lang]
            if show_std:
                row_values.append(f"{corr:.4f}±{std:.4f}")
            else:
                row_values.append(f"{corr:.4f}")
        print("\t".join(languages))
        print("\t".join(row_values))
        
        # Tab-separated values for easy copy-paste
        print("\nKendall's Tau-b:")
        row_values = []
        for lang in languages:
            corr, std = kendall_results[lang]
            if show_std:
                row_values.append(f"{corr:.4f}±{std:.4f}")
            else:
                row_values.append(f"{corr:.4f}")
        print("\t".join(languages))
        print("\t".join(row_values))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    #parser.add_argument("predictions_path_prefix", type=Path)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--show-std", action="store_true", help="Show standard deviation in results")
    parser.add_argument("--split-annotators", action="store_true", help="Compute correlations separately for each annotator")
    args = parser.parse_args()



    main(args.short if args.short else False, args.show_std, args.split_annotators)