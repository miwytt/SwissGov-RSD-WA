import json
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import argparse as ap
from difflib import SequenceMatcher
import numpy as np


def get_f1(file1_data, file2_data):
    """
    Calculate the F1 score for a text pair – each language separately.
    F1 is the harmonic mean of the precision and recall.
    """
    def f1(pred, gold):
        pred_set = {tuple(pos) for item in pred for pos in item}
        gold_set = {tuple(pos) for item in gold for pos in item}

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    file1_text1_positions = [item["text1_positions"] for item in file1_data]
    file2_text1_positions = [item["text1_positions"] for item in file2_data]
    file1_text2_positions = [item["text2_positions"] for item in file1_data]
    file2_text2_positions = [item["text2_positions"] for item in file2_data]

    f1_other = f1(file1_text1_positions, file2_text1_positions)
    f1_en = f1(file1_text2_positions, file2_text2_positions)

    return f1_en, f1_other


def get_iou(file1_data, file2_data):
    """
    Calculate the Intersection over Union (IOU) for a text pair – each language separately.
    IOU is the ratio of the intersection of the two sets of positions to the union of the two sets of positions.
    """
    file1_positions_other = {tuple(pos) for item in file1_data for pos in item["text1_positions"]}
    file2_positions_other = {tuple(pos) for item in file2_data for pos in item["text1_positions"]}
    file1_positions_en = {tuple(pos) for item in file1_data for pos in item["text2_positions"]}
    file2_positions_en = {tuple(pos) for item in file2_data for pos in item["text2_positions"]}

    iou_en = len(file1_positions_en.intersection(file2_positions_en)) / len(file1_positions_en.union(file2_positions_en)) if len(file1_positions_en.union(file2_positions_en)) else 0
    iou_other = len(file1_positions_other.intersection(file2_positions_other)) / len(file1_positions_other.union(file2_positions_other)) if len(file1_positions_other.union(file2_positions_other)) else 0
    return iou_en, iou_other


def get_match_spearman(file1_data, file2_data):
    """
    Calculate the Spearman's rank correlation coefficient between the labels for all spans of two annotators that match exactly.
    """
    file1_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) for item in file1_data}
    file2_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) for item in file2_data}

    # find all items in file1_data whose text1_positions are in items of file2_data and vice versa
    file1_scores = [item['score'] for item in file1_data if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) in file2_lookup ]
    file2_scores = [item['score'] for item in file2_data if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) in file1_lookup ]

    num_tokens_file1 = sum(len(item["text1"]) for item in file1_data) + sum(len(item["text2"]) for item in file1_data)
    num_tokens_file2 = sum(len(item["text1"]) for item in file2_data) + sum(len(item["text2"]) for item in file2_data)

    file1_items = [item for item in file1_data if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) in file2_lookup ]
    file2_items = [item for item in file2_data if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"])), item["pair"]) in file1_lookup ]

    
    # calculate the Spearman's rank correlation coefficient for the two files for all spans that match exactly
    match_spearman = spearmanr(file1_scores, file2_scores)

    number_of_matches = len(file1_scores)

    return match_spearman, number_of_matches, num_tokens_file1, num_tokens_file2


def get_fuzzy_spearman(file1_data, file2_data):
    """
    Calculate the Spearman's rank correlation coefficient between the labels for all spans of two annotators that match exactly but with an IoU threshold of 0.5, 0.75, 0.90.
    """
    data_of_same_files = [(item1, item2) for item1 in file1_data for item2 in file2_data if item1["pair"] == item2["pair"]]

    # find all items in file1_data whose text1_positions are in items of file2_data and vice versa but not as exact match but with an IoU threshold of 0.5, 0.75, 0.90
    file1_scores_5 = []
    file2_scores_5 = []
    file1_scores_75 = []
    file2_scores_75 = []
    file1_scores_90 = []
    file2_scores_90 = []

    for item1, item2 in data_of_same_files:
        text1_pos1 = set(map(tuple, item1["text1_positions"]))
        text1_pos2 = set(map(tuple, item2["text1_positions"]))
        text2_pos1 = set(map(tuple, item1["text2_positions"]))
        text2_pos2 = set(map(tuple, item2["text2_positions"]))

        overlap_text1 = text1_pos1.intersection(text1_pos2)
        overlap_text2 = text2_pos1.intersection(text2_pos2)

        if overlap_text1 or overlap_text2:
            union_text1 = text1_pos1.union(text1_pos2)
            union_text2 = text2_pos1.union(text2_pos2)

            iou_en = len(overlap_text1) / len(union_text1) if union_text1 else 1.0
            iou_other = len(overlap_text2) / len(union_text2) if union_text2 else 1.0

            iou = (iou_en + iou_other) / 2

            if iou >= 0.90:
                file1_scores_90.append(item1["score"])
                file2_scores_90.append(item2["score"])
            
            if iou >= 0.75:
                file1_scores_75.append(item1["score"])
                file2_scores_75.append(item2["score"])
            
            if iou >= 0.5:
                file1_scores_5.append(item1["score"])
                file2_scores_5.append(item2["score"])
    
    assert len(file1_scores_5) == len(file2_scores_5) 
    assert len(file1_scores_5) >= len(file2_scores_75) >= len(file2_scores_90)

    # calculate the Spearman's rank correlation coefficient for the two files for all spans that match exactly but with an IoU threshold of 0.5, 0.75, 0.90
    match_spearman_5 = spearmanr(file1_scores_5, file2_scores_5)
    match_spearman_75 = spearmanr(file1_scores_75, file2_scores_75)
    match_spearman_90 = spearmanr(file1_scores_90, file2_scores_90)

    number_of_matches_5 = len(file1_scores_5)
    number_of_matches_75 = len(file1_scores_75)
    number_of_matches_90 = len(file1_scores_90)

    return match_spearman_5, match_spearman_75, match_spearman_90, number_of_matches_5, number_of_matches_75, number_of_matches_90


def get_character_overlap_spearman(file1_data, file2_data):
    """Use character-level overlap for fuzzy matching"""
    data_of_same_files = [(item1, item2) for item1 in file1_data for item2 in file2_data if item1["pair"] == item2["pair"]]

    print(f"Number of data of same files: {len(data_of_same_files)}")
    print(f"Number of data of same files: {len(file1_data)}")
    print(f"Number of data of same files: {len(file2_data)}")

    file1_scores_high = []
    file2_scores_high = []
    file1_scores_medium = []
    file2_scores_medium = []
    file1_scores_low = []
    file2_scores_low = []

    num_low_pairs = 0
    num_medium_pairs = 0
    num_high_pairs = 0
    

    for item1, item2 in data_of_same_files:
        # Get the actual text spans
        text1_span1 = item1["text1"]
        text1_span2 = item2["text1"]
        text2_span1 = item1["text2"]
        text2_span2 = item2["text2"]
        
        # Calculate character-level similarity using difflib
        
        similarity_text1 = SequenceMatcher(None, text1_span1, text1_span2).ratio()
        similarity_text2 = SequenceMatcher(None, text2_span1, text2_span2).ratio()
        
        avg_similarity = (similarity_text1 + similarity_text2) / 2
        
        if avg_similarity >= 0.5:
            file1_scores_low.append(item1["score"])
            file2_scores_low.append(item2["score"])
            num_low_pairs += 1
        if avg_similarity >= 0.75:
            file1_scores_medium.append(item1["score"])
            file2_scores_medium.append(item2["score"])
            num_medium_pairs += 1
        if avg_similarity >= 0.9:
            file1_scores_high.append(item1["score"])
            file2_scores_high.append(item2["score"])
            num_high_pairs += 1

    low_spearman = spearmanr(file1_scores_low, file2_scores_low)
    medium_spearman = spearmanr(file1_scores_medium, file2_scores_medium)
    high_spearman = spearmanr(file1_scores_high, file2_scores_high)

    return low_spearman, medium_spearman, high_spearman, num_low_pairs, num_medium_pairs, num_high_pairs


def get_pair_spearman_rankings(file1_data, file2_data):
    """
    Calculate Spearman correlation for each individual annotation pair and return 
    the top 5 and bottom 5 pairs by correlation.
    """
    # Group data by pair ID
    file1_by_pair = {}
    file2_by_pair = {}
    
    for item in file1_data:
        pair_id = item["pair"]
        if pair_id not in file1_by_pair:
            file1_by_pair[pair_id] = []
        file1_by_pair[pair_id].append(item)
    
    for item in file2_data:
        pair_id = item["pair"]
        if pair_id not in file2_by_pair:
            file2_by_pair[pair_id] = []
        file2_by_pair[pair_id].append(item)
    
    # Calculate Spearman correlation for each pair
    pair_correlations = []
    
    for pair_id in file1_by_pair:
        if pair_id in file2_by_pair:
            # Get matching items between the two annotators for this pair
            file1_items = file1_by_pair[pair_id]
            file2_items = file2_by_pair[pair_id]
            
            # Find exact matches based on text positions
            file1_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) for item in file1_items}
            file2_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) for item in file2_items}
            
            # Get matching scores
            file1_scores = [item['score'] for item in file1_items if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) in file2_lookup]
            file2_scores = [item['score'] for item in file2_items if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) in file1_lookup]
            
            # Only calculate correlation if we have at least 2 matching items
            if len(file1_scores) >= 2 and len(file2_scores) >= 2:
                # Check if either array is constant (all values the same)
                if len(set(file1_scores)) == 1 or len(set(file2_scores)) == 1:
                    print(f"  Skipping pair {pair_id}: constant values detected (file1: {len(set(file1_scores))} unique values, file2: {len(set(file2_scores))} unique values)")
                    continue
                
                try:
                    correlation = spearmanr(file1_scores, file2_scores)
                    if not np.isnan(correlation.statistic):  # Check for valid correlation
                        pair_correlations.append({
                            'pair_id': pair_id,
                            'correlation': correlation.statistic,
                            'p_value': correlation.pvalue,
                            'num_matches': len(file1_scores)
                        })
                    else:
                        print(f"  Skipping pair {pair_id}: correlation is NaN")
                except Exception as e:
                    print(f"  Skipping pair {pair_id}: correlation calculation error - {e}")
                    continue
    
    # Sort by correlation
    pair_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    print(f"Total pairs with valid correlations: {len(pair_correlations)}")
    
    # Debug: Show some statistics about the data
    if len(pair_correlations) == 0:
        print("Debug: No valid correlations found. Let's check the data...")
        print(f"  Total pairs processed: {len(file1_by_pair)}")
        print(f"  Pairs with matches in both files: {len([p for p in file1_by_pair if p in file2_by_pair])}")
        
        # Check a few examples
        example_count = 0
        for pair_id in list(file1_by_pair.keys())[:3]:  # Check first 3 pairs
            if pair_id in file2_by_pair:
                file1_items = file1_by_pair[pair_id]
                file2_items = file2_by_pair[pair_id]
                
                file1_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) for item in file1_items}
                file2_lookup = {(tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) for item in file2_items}
                
                file1_scores = [item['score'] for item in file1_items if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) in file2_lookup]
                file2_scores = [item['score'] for item in file2_items if (tuple(map(tuple, item["text1_positions"])), tuple(map(tuple, item["text2_positions"]))) in file1_lookup]
                
                print(f"  Example pair {pair_id}:")
                print(f"    File1 scores: {file1_scores[:10]} (unique values: {len(set(file1_scores))})")
                print(f"    File2 scores: {file2_scores[:10]} (unique values: {len(set(file2_scores))})")
                print(f"    Match count: {len(file1_scores)}")
                example_count += 1
                if example_count >= 3:
                    break
    
    # Get top 5 and bottom 5
    # Top 5: highest correlations (first 5 in sorted list)
    top_5 = pair_correlations[:5] if len(pair_correlations) >= 5 else pair_correlations
    
    # Bottom 5: lowest correlations (last 5 in sorted list, but reverse order for display)
    if len(pair_correlations) >= 10:
        # If we have 10+ pairs, get the last 5 and reverse them for display
        bottom_5 = pair_correlations[-5:][::-1]
    elif len(pair_correlations) >= 5:
        # If we have 5-9 pairs, get the last few and reverse them
        bottom_5 = pair_correlations[5:][::-1]
    else:
        # If we have fewer than 5 pairs, no bottom 5 to show
        bottom_5 = []
    
    print(f"Returning {len(top_5)} top pairs and {len(bottom_5)} bottom pairs")
    
    return top_5, bottom_5


def get_label_distribution(data1, data2, granularity='annotators'):
    if granularity == 'annotators':
        label_distribution1 = {}
        label_distribution2 = {}
        for item in data1:
            if item['score'] not in label_distribution1:
                label_distribution1[item['score']] = 1
            label_distribution1[item['score']] += 1
        for item in data2:
            if item['score'] not in label_distribution2:
                label_distribution2[item['score']] = 1
            label_distribution2[item['score']] += 1
        return label_distribution1, label_distribution2
    """
    elif granularity == 'languages':
        pass
    """
    


def load_jsonl(file_path):
    """Load a JSONL file and return a list of JSON objects."""
    data = []
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: JSON decode error in {file_path} at line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"  Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"  Error reading file {file_path}: {e}")
        return []
    return data


def main():
    """de_data1 = []
    de_data2 = []
    fr_data1 = []
    fr_data2 = []
    it_data1 = []
    it_data2 = []"""

    parser = ap.ArgumentParser()
    parser.add_argument("input_folder1", type=str, help="Path to input folder of containing first set of annotations.")
    parser.add_argument("input_folder2", type=str, help="Path to input folder of containing second set of annotations to compare to.")
    parser.add_argument("--trial", action="store_true", help="Whether to evaluate the trial annotations.")
    args = parser.parse_args()


    # Store results for all languages to create combined table at the end
    all_results = {}


    num_low_pairs_total_de = 0
    num_medium_pairs_total_de = 0
    num_high_pairs_total_de = 0
    num_exact_matches_total_de = 0

    num_low_pairs_total_fr = 0
    num_medium_pairs_total_fr = 0
    num_high_pairs_total_fr = 0
    num_exact_matches_total_fr = 0

    num_low_pairs_total_it = 0
    num_medium_pairs_total_it = 0
    num_high_pairs_total_it = 0
    num_exact_matches_total_it = 0



    
    for lang in ["de", "fr", "it"]:
        total_iou_en = []
        total_iou_other = []
        total_f1_en = []
        total_f1_other = []
        spearman_data_1 = []
        spearman_data_2 = []

        num_differences_total_annotator1 = 0
        num_differences_total_annotator2 = 0

        total_number_of_matches = 0

        total_number_of_matches_5 = 0
        total_number_of_matches_75 = 0
        total_number_of_matches_90 = 0

        num_tokens_total_file1 = 0
        num_tokens_total_file2 = 0
        
        print(f"Input folder 1: {args.input_folder1}")
        print(f"Input folder 2: {args.input_folder2}")
        print(f"Language: {lang}")
        print(f"Number of files in input folder 1: {len(os.listdir(f'{args.input_folder1}/{lang}'))}")
        print(f"Number of files in input folder 2: {len(os.listdir(f'{args.input_folder2}/{lang}'))}")
        assert len(os.listdir(f"{args.input_folder1}/{lang}")) == len(os.listdir(f"{args.input_folder2}/{lang}"))  

        print(f"Language Pair: EN-{lang.upper()}")

        overlap_files = []
        for file in os.listdir(f"{args.input_folder1}/{lang}"):
            if file in os.listdir(f"{args.input_folder2}/{lang}"):
                overlap_files.append(file)

        print(f"Number of overlap files: {len(overlap_files)}")

        processed_files = 0
        skipped_files = 0

        for file_o in overlap_files:
            print(f"\nProcessing file: {file_o}")
            
            file1_path = f"{args.input_folder1}/{lang}/{file_o}"
            file2_path = f"{args.input_folder2}/{lang}/{file_o}"

            file1_data = load_jsonl(file1_path) 
            file2_data = load_jsonl(file2_path)
            
            print(f"  File1 data points: {len(file1_data)}")
            print(f"  File2 data points: {len(file2_data)}") 

            # Check if file has data before asserting
            if not file1_data:
                print(f"\n{os.path.basename(file1_path)} has no data - skipping")
                skipped_files += 1
                continue
            if not file2_data:
                print(f"\n{os.path.basename(file2_path)} has no data - skipping")
                skipped_files += 1
                continue

            assert "text1" in file1_data[0]
            assert "text2" in file1_data[0]
            assert "score" in file1_data[0]
            assert "page_other" in file1_data[0]
            assert "page_en" in file1_data[0]
            assert "languages" in file1_data[0]
            assert "pair" in file1_data[0]

            assert "text1" in file2_data[0]
            assert "text2" in file2_data[0]
            assert "score" in file2_data[0]
            assert "page_other" in file2_data[0]
            assert "page_en" in file2_data[0]
            assert "languages" in file2_data[0]
            assert "pair" in file2_data[0]

            if not "text1_positions" in file2_data[0] and not "text2_positions" in file2_data[0]:
                print(f"\n{os.path.basename(file2_path)} has no text position indicators")
                continue # TODO: implement handling

            if not "text1_positions" in file1_data[0] and not "text2_positions" in file1_data[0]:
                print(f"\n{os.path.basename(file1_path)} has no text position indicators")
                continue # TODO: implement handling

            # Check for position fields if they exist
            for i, item in enumerate(file1_data):
                if "text1_positions" in item and "text2_positions" in item:
                    text1_len = len(item["text1"])
                    text1_pos_len = len(item["text1_positions"])
                    if text1_len != text1_pos_len:
                        print(f"\nrow {i+1}: text1 length: {text1_len}, text1_positions length: {text1_pos_len}")
                    assert text1_len == text1_pos_len
                    assert len(item["text2"]) == len(item["text2_positions"])

            # Check for position fields if they exist
            for i, item in enumerate(file2_data):
                if "text1_positions" in item and "text2_positions" in item:
                    text2_len = len(item["text2"])
                    text2_pos_len = len(item["text2_positions"])
                    if text2_len != text2_pos_len:
                        print(f"\nrow {i+1}: text2 length: {text2_len}, text2_positions length: {text2_pos_len}")
                    assert text2_len == text2_pos_len
                    assert len(item["text1"]) == len(item["text1_positions"])

            num_differences_file1 = len(file1_data)
            num_differences_file2 = len(file2_data)

            """print(f"\nNumber of differences {os.path.basename(file1_path)}: {num_differences_file1}")
            print(f"Number of differences {os.path.basename(file2_path)}: {num_differences_file2}")"""
            num_differences_total_annotator1 += num_differences_file1
            num_differences_total_annotator2 += num_differences_file2

            iou_en, iou_other = get_iou(file1_data, file2_data)
            total_iou_en.append(iou_en)
            total_iou_other.append(iou_other)
            print()
            print(f"IOU EN: {iou_en:.2f}")
            print(f"IOU OTHER: {iou_other:.2f}")

            f1_en, f1_other = get_f1(file1_data, file2_data)
            total_f1_en.append(f1_en)
            total_f1_other.append(f1_other)
            print(f"F1 EN: {f1_en:.2f}")
            print(f"F1 OTHER: {f1_other:.2f}")

            spearman_data_1.extend(file1_data)
            spearman_data_2.extend(file2_data)
            processed_files += 1

        # since spearman correlation is only meaningful for sample size > 500, it does not make sense to compute file level correlation
        spearman_exact_match, number_of_matches, num_tokens_file1, num_tokens_file2 = get_match_spearman(spearman_data_1, spearman_data_2)
        spearman_fuzzy_match_5, spearman_fuzzy_match_75, spearman_fuzzy_match_90, number_of_matches_5, number_of_matches_75, number_of_matches_90 = get_fuzzy_spearman(spearman_data_1, spearman_data_2)
        spearman_character_overlap_low, spearman_character_overlap_medium, spearman_character_overlap_high, num_low_pairs, num_medium_pairs, num_high_pairs = get_character_overlap_spearman(spearman_data_1, spearman_data_2)
        
        # Get top 5 and bottom 5 pair correlations
        top_5_pairs, bottom_5_pairs = get_pair_spearman_rankings(spearman_data_1, spearman_data_2)
        
        print(f"Found {len(top_5_pairs)} top pairs and {len(bottom_5_pairs)} bottom pairs for {lang.upper()}")
        
        print(f"Number of matches: {number_of_matches}")
        total_number_of_matches += number_of_matches

        num_tokens_total_file1 += num_tokens_file1
        num_tokens_total_file2 += num_tokens_file2


        total_number_of_matches_5 += number_of_matches_5
        total_number_of_matches_75 += number_of_matches_75
        total_number_of_matches_90 += number_of_matches_90

        if lang == "de":
            num_exact_matches_total_de += number_of_matches
            num_low_pairs_total_de += number_of_matches_5
            num_medium_pairs_total_de += number_of_matches_75
            num_high_pairs_total_de += number_of_matches_90 
        elif lang == "fr":
            num_exact_matches_total_fr += number_of_matches
            num_low_pairs_total_fr += number_of_matches_5
            num_medium_pairs_total_fr += number_of_matches_75
            num_high_pairs_total_fr += number_of_matches_90
        elif lang == "it":  
            num_exact_matches_total_it += number_of_matches
            num_low_pairs_total_it += number_of_matches_5
            num_medium_pairs_total_it += number_of_matches_75
            num_high_pairs_total_it += number_of_matches_90

        # label distribution
        label_distribution1, label_distribution2 = get_label_distribution(spearman_data_1, spearman_data_2, granularity='annotators')

        # Store results for this language
        all_results[lang] = {
            'iou_en': (sum(total_iou_en) / len(total_iou_en)) * 100,
            'iou_other': (sum(total_iou_other) / len(total_iou_other)) * 100,
            'f1_en': (sum(total_f1_en) / len(total_f1_en)) * 100,
            'f1_other': (sum(total_f1_other) / len(total_f1_other)) * 100,
            'spearman_exact': spearman_exact_match.statistic * 100,
            'spearman_5': spearman_fuzzy_match_5.statistic * 100,
            'spearman_75': spearman_fuzzy_match_75.statistic * 100,
            'spearman_90': spearman_fuzzy_match_90.statistic * 100,
            'spearman_character_overlap_low': spearman_character_overlap_low.statistic * 100,
            'spearman_character_overlap_medium': spearman_character_overlap_medium.statistic * 100,
            'spearman_character_overlap_high': spearman_character_overlap_high.statistic * 100
        }

        print(f"\nP-values:")
        print(f"Exact Match Spearman: {spearman_exact_match.pvalue:.3f}")
        print(f"Fuzzy Match Spearman 0.5: {spearman_fuzzy_match_5.pvalue:.3f}")
        print(f"Fuzzy Match Spearman 0.75: {spearman_fuzzy_match_75.pvalue:.3f}")
        print(f"Fuzzy Match Spearman 0.90: {spearman_fuzzy_match_90.pvalue:.3f}")
        print(f"Character Overlap Spearman 0.5: {spearman_character_overlap_low.pvalue:.3f}")
        print(f"Character Overlap Spearman 0.75: {spearman_character_overlap_medium.pvalue:.3f}")
        print(f"Character Overlap Spearman 0.90: {spearman_character_overlap_high.pvalue:.3f}")
        print()
        
        # Print top 5 and bottom 5 pair correlations
        print(f"\nTop 5 Annotation Pairs by Spearman Correlation ({lang.upper()}):")
        print(f"{'Rank':<5} | {'Pair ID':<15} | {'Correlation':<12} | {'P-value':<10} | {'Matches':<8}")
        print("-" * 70)
        for i, pair in enumerate(top_5_pairs, 1):
            print(f"{i:<5} | {pair['pair_id']:<15} | {pair['correlation']:<12.4f} | {pair['p_value']:<10.4f} | {pair['num_matches']:<8}")
        
        if bottom_5_pairs:
            print(f"\nBottom 5 Annotation Pairs by Spearman Correlation ({lang.upper()}):")
            print(f"{'Rank':<5} | {'Pair ID':<15} | {'Correlation':<12} | {'P-value':<10} | {'Matches':<8}")
            print("-" * 70)
            for i, pair in enumerate(bottom_5_pairs, 1):
                print(f"{i:<5} | {pair['pair_id']:<15} | {pair['correlation']:<12.4f} | {pair['p_value']:<10.4f} | {pair['num_matches']:<8}")
        else:
            print(f"\nNo bottom 5 pairs available for {lang.upper()} (insufficient data)")
        print()
        
        # Save pair correlations to CSV
        """os.makedirs("stats", exist_ok=True)
        path = "stats/trial" if args.trial else "stats"
        with open(f"{path}/{lang.upper()}_pair_correlations.csv", "w") as f:
            f.write("Rank,Pair_ID,Correlation,P_value,Num_Matches,Category\n")
            for i, pair in enumerate(top_5_pairs, 1):
                f.write(f"{i},{pair['pair_id']},{pair['correlation']:.4f},{pair['p_value']:.4f},{pair['num_matches']},Top\n")
            for i, pair in enumerate(bottom_5_pairs, 1):
                f.write(f"{i},{pair['pair_id']},{pair['correlation']:.4f},{pair['p_value']:.4f},{pair['num_matches']},Bottom\n")"""

        """# save label distributions as csv and print
        os.makedirs("stats", exist_ok=True)
        path = "stats/trial" if args.trial else "stats"
        with open(f"{path}/{lang.upper()}_label_distribution.csv", "w") as f:
            f.write("Score,Annotator 1,Annotator 2\n")
            print(f"{lang.upper()} Label Distribution:")
            print(f"{'Score':<10} | {'Annotator 1':<10} | {'Annotator 2':<10}")
            print("-" * 30)
            for score in sorted(label_distribution1.keys()):
                f.write(f"{score},{label_distribution1[score]},{label_distribution2[score]}\n")
                print(f"{score:<10} | {label_distribution1[score]:<10} | {label_distribution2[score]:<10}")
        print()

        # Store data for each language to plot together at the end
        if not hasattr(main, 'plot_data'):
            main.plot_data = {}
        main.plot_data[lang] = {
            'values1': [label_distribution1.get(score, 0) for score in range(1,6)],
            'values2': [label_distribution2.get(score, 0) for score in range(1,6)]
        }

        print(f"Number of differences total annotator 1: {num_differences_total_annotator1}")
        print(f"Number of differences total annotator 2: {num_differences_total_annotator2}")

        print(f"Number of exact matches total: {total_number_of_matches}")

        print(f"Number of matches 5: {total_number_of_matches_5}")
        print(f"Number of matches 75: {total_number_of_matches_75}")
        print(f"Number of matches 90: {total_number_of_matches_90}")

        print(f"Number of tokens total annotator 1: {num_tokens_total_file1}")
        print(f"Number of tokens total annotator 2: {num_tokens_total_file2}")

        print(f"\nFile processing summary for {lang.upper()}:")
        print(f"  Total overlap files: {len(overlap_files)}")
        print(f"  Successfully processed: {processed_files}")
        print(f"  Skipped (empty/invalid): {skipped_files}")

        # After processing all languages, create combined plot
        if lang == 'it':  # Last language processed
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            width = 0.35
            x = [1, 2, 3, 4, 5]
            x_pos = [i - width/2 for i in range(len(x))]

            # Plot DE data
            ax1.bar(x_pos, main.plot_data['de']['values1'], width, label='Annotator 1')
            ax1.bar([i + width/2 for i in range(len(x))], main.plot_data['de']['values2'], width, label='Annotator 2')
            ax1.set_xticks(range(len(x)))
            ax1.set_xticklabels(x)
            ax1.set_title('German')
            ax1.set_ylabel('Number of Differences')
            ax1.legend()

            # Plot FR data  
            ax2.bar(x_pos, main.plot_data['fr']['values1'], width, label='Annotator 1')
            ax2.bar([i + width/2 for i in range(len(x))], main.plot_data['fr']['values2'], width, label='Annotator 2')
            ax2.set_xticks(range(len(x)))
            ax2.set_xticklabels(x)
            ax2.set_title('French')
            ax2.set_xlabel('Difference Labels')
            ax2.legend()

            # Plot IT data
            ax3.bar(x_pos, main.plot_data['it']['values1'], width, label='Annotator 1')
            ax3.bar([i + width/2 for i in range(len(x))], main.plot_data['it']['values2'], width, label='Annotator 2')
            ax3.set_xticks(range(len(x)))
            ax3.set_xticklabels(x)
            ax3.set_title('Italian')
            ax3.legend()

            plt.tight_layout()
            os.makedirs("visualizations", exist_ok=True)
            path = "visualizations/trial" if args.trial else "visualizations"
            plt.savefig(f"{path}/all_languages_label_distribution.svg")
            plt.savefig(f"{path}/all_languages_label_distribution.png")
            plt.close()


        print("-" * 100)"""



    # Print combined table for all languages
    print()
    print("SUMMARY TABLE:")
    print()
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Metric} & \\textbf{DE} & \\textbf{FR} & \\textbf{IT} \\\\")
    print("\\hline")
    print(f"Mean IOU EN & {all_results['de']['iou_en']:.2f} & {all_results['fr']['iou_en']:.2f} & {all_results['it']['iou_en']:.2f} \\\\")
    print(f"Mean IOU OTHER & {all_results['de']['iou_other']:.2f} & {all_results['fr']['iou_other']:.2f} & {all_results['it']['iou_other']:.2f} \\\\")
    print(f"Mean F1 EN & {all_results['de']['f1_en']:.2f} & {all_results['fr']['f1_en']:.2f} & {all_results['it']['f1_en']:.2f} \\\\")
    print(f"Mean F1 OTHER & {all_results['de']['f1_other']:.2f} & {all_results['fr']['f1_other']:.2f} & {all_results['it']['f1_other']:.2f} \\\\")
    print(f"Exact Match Spearman & {all_results['de']['spearman_exact']:.2f} & {all_results['fr']['spearman_exact']:.2f} & {all_results['it']['spearman_exact']:.2f} \\\\")
    print(f"Fuzzy Match Spearman 0.5 & {all_results['de']['spearman_5']:.2f} & {all_results['fr']['spearman_5']:.2f} & {all_results['it']['spearman_5']:.2f} \\\\")
    print(f"Fuzzy Match Spearman 0.75 & {all_results['de']['spearman_75']:.2f} & {all_results['fr']['spearman_75']:.2f} & {all_results['it']['spearman_75']:.2f} \\\\")
    print(f"Fuzzy Match Spearman 0.90 & {all_results['de']['spearman_90']:.2f} & {all_results['fr']['spearman_90']:.2f} & {all_results['it']['spearman_90']:.2f} \\\\")
    print(f"Character Overlap Spearman 0.5 & {all_results['de']['spearman_character_overlap_low']:.2f} & {all_results['fr']['spearman_character_overlap_low']:.2f} & {all_results['it']['spearman_character_overlap_low']:.2f} \\\\")
    print(f"Character Overlap Spearman 0.75 & {all_results['de']['spearman_character_overlap_medium']:.2f} & {all_results['fr']['spearman_character_overlap_medium']:.2f} & {all_results['it']['spearman_character_overlap_medium']:.2f} \\\\")
    print(f"Character Overlap Spearman 0.90 & {all_results['de']['spearman_character_overlap_high']:.2f} & {all_results['fr']['spearman_character_overlap_high']:.2f} & {all_results['it']['spearman_character_overlap_high']:.2f} \\\\")
    print(f"Span pairs with similarity >= 0.5 & {num_low_pairs_total_de} & {num_low_pairs_total_fr} & {num_low_pairs_total_it} \\\\")
    print(f"Span pairs with similarity >= 0.75 & {num_medium_pairs_total_de} & {num_medium_pairs_total_fr} & {num_medium_pairs_total_it} \\\\")
    print(f"Span pairs with similarity >= 0.90 & {num_high_pairs_total_de} & {num_high_pairs_total_fr} & {num_high_pairs_total_it} \\\\")
    print(f"Exact matches & {num_exact_matches_total_de} & {num_exact_matches_total_fr} & {num_exact_matches_total_it} \\\\")
    print("\\hline")
    print("\\end{tabular}")


    print()
    print("VISUALIZATIONS HAVE BEEN SAVED IN THE VISUALIZATIONS FOLDER")
    print()

        

if __name__ == "__main__":
    main()