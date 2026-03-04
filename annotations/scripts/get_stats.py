import jsonlines
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import argparse as ap

# Initialize nested defaultdict to avoid manual initialization of scores
from collections import defaultdict
label_distribution = defaultdict(lambda: defaultdict(int))
        

def main():

    parser = ap.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to input folder of containing the final annotations.")
    parser.add_argument("--ists", action="store_true", help="for ists dataset.")
    args = parser.parse_args()

    with open("list_to_drop.txt", "r") as f:
        list_to_drop = f.read().splitlines()
        list_to_drop = [_.strip() for _ in list_to_drop]

    # total num tokens per language
    total_num_tokens = defaultdict(int)

    if args.ists:
        for lang in ["de", "fr", "it"]:
            previous_min_length = None
            previous_max_length = None
            total_length_b = 0  # For target language
            total_length_a = 0  # For English
            num_items = 0
            
            with jsonlines.open(f"{args.input_folder}/gold.jsonl") as reader:
                for item in reader:

                    if lang not in item['id']:
                        continue

                    total_num_tokens[lang] += len(item['labels_b'])
                    total_num_tokens[f"en_{lang}"] += len(item['labels_a'])

                    # Track total lengths for average calculation
                    total_length_b += len(item['labels_b'])
                    total_length_a += len(item['labels_a'])
                    num_items += 1

                    min_length = min(len(item['labels_b']), len(item['labels_a']))
                    if previous_min_length is None:
                        previous_min_length = min_length
                    else:
                        if min_length < previous_min_length:
                            previous_min_length = min_length
                            #print(f"New min length: {previous_min_length}")

                    max_length = max(len(item['labels_b']), len(item['labels_a']))
                    if previous_max_length is None:
                        previous_max_length = max_length
                    else:
                        if max_length > previous_max_length:
                            previous_max_length = max_length
                            #print(f"New max length: {previous_max_length}")

                    for score in [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                        label_distribution[lang][score] += item['labels_b'].count(score)
                        label_distribution[f"en_{lang}"][score] += item['labels_a'].count(score)

            # Assert total tokens matches sum of labeled tokens
            """assert total_num_tokens[lang] == sum(label_distribution[lang].values()), \
                f"Token count mismatch for {lang}: total={total_num_tokens[lang]}, sum={sum(label_distribution[lang].values())}"
            assert total_num_tokens[f"en_{lang}"] == sum(label_distribution[f"en_{lang}"].values()), \
                f"Token count mismatch for en_{lang}: total={total_num_tokens[f'en_{lang}']}, sum={sum(label_distribution[f'en_{lang}'].values())}"
            """
        
            print(f"Min length for {lang}: {previous_min_length}")
            print(f"Max length for {lang}: {previous_max_length}")
            print(f"Average length for {lang}: {total_length_b/num_items:.2f}")
            print(f"Average length for en_{lang}: {total_length_a/num_items:.2f}")
            print(f"Average of both for {lang}: {(total_length_b + total_length_a)/(2*num_items):.2f}")
    
    else:

        for lang in ["de", "fr", "it"]:
            previous_min_length = None
            previous_max_length = None
            total_length_b = 0  # For target language
            total_length_a = 0  # For English
            num_items = 0
            
            with jsonlines.open(f"{args.input_folder}/gold_admin_{lang}.jsonl") as reader:
                for item in reader:

                    """if item['id'] in list_to_drop:
                        continue"""

                    total_num_tokens[lang] += len(item['labels_b'])
                    total_num_tokens[f"en_{lang}"] += len(item['labels_a'])

                    # Track total lengths for average calculation
                    total_length_b += len(item['labels_b'])
                    total_length_a += len(item['labels_a'])
                    num_items += 1

                    min_length = min(len(item['labels_b']), len(item['labels_a']))
                    if previous_min_length is None:
                        previous_min_length = min_length
                    else:
                        if min_length < previous_min_length:
                            previous_min_length = min_length
                            #print(f"New min length: {previous_min_length}")

                    max_length = max(len(item['labels_b']), len(item['labels_a']))
                    if previous_max_length is None:
                        previous_max_length = max_length
                    else:
                        if max_length > previous_max_length:
                            previous_max_length = max_length
                            #print(f"New max length: {previous_max_length}")

                    for score in [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                        # round scores in item['labels_b'] and item['labels_a'] to 1 decimal place
                        item['labels_b'] = [round(score, 1) for score in item['labels_b']]
                        item['labels_a'] = [round(score, 1) for score in item['labels_a']]
                        label_distribution[lang][score] += item['labels_b'].count(score)
                        label_distribution[f"en_{lang}"][score] += item['labels_a'].count(score)

            # Assert total tokens matches sum of labeled tokens
            assert total_num_tokens[lang] == sum(label_distribution[lang].values()), \
                f"Token count mismatch for {lang}: total={total_num_tokens[lang]}, sum={sum(label_distribution[lang].values())}"
            assert total_num_tokens[f"en_{lang}"] == sum(label_distribution[f"en_{lang}"].values()), \
                f"Token count mismatch for en_{lang}: total={total_num_tokens[f'en_{lang}']}, sum={sum(label_distribution[f'en_{lang}'].values())}"

        
            print(f"Min length for {lang}: {previous_min_length}")
            print(f"Max length for {lang}: {previous_max_length}")
            print(f"Average length for {lang}: {total_length_b/num_items:.2f}")
            print(f"Average length for en_{lang}: {total_length_a/num_items:.2f}")
            print(f"Average of both for {lang}: {(total_length_b + total_length_a)/(2*num_items):.2f}")
    
    
    # Print total token counts
    print("\nTotal Token Counts:")
    print("-" * 50)
    print(f"{'Language':<15} {'Count':<10}")
    print("-" * 50)
    for lang, count in total_num_tokens.items():
        print(f"{lang:<15} {count:<10}")
    print("-" * 50)

    # Print non-English label distribution
    print("\nNon-English Label Distribution:")
    print("-" * 50)
    print(f"{'Language':<15} {'-1.0':<8} {'0.0':<8} {'0.2':<8} {'0.4':<8} {'0.6':<8} {'0.8':<8} {'1.0':<8}")
    print("-" * 50)
    for lang in ["de", "fr", "it"]:
        row = [str(label_distribution[lang][score]) for score in [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        print(f"{lang:<15} {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    print("-" * 50)

    # Print English label distribution
    print("\nEnglish Label Distribution:")
    print("-" * 50)
    print(f"{'Language':<15} {'-1.0':<8} {'0.0':<8} {'0.2':<8} {'0.4':<8} {'0.6':<8} {'0.8':<8} {'1.0':<8}")
    print("-" * 50)
    for lang in ["en_de", "en_fr", "en_it"]:
        row = [str(label_distribution[lang][score]) for score in [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        print(f"{lang:<15} {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    print("-" * 50)


if __name__ == "__main__":
    main()