import json
from pathlib import Path
import sys
import os
import argparse as ap

import jsonlines
import json
import jinja2

parser = ap.ArgumentParser()
parser.add_argument("--lang", type=str, help="Language to use for test data")
parser.add_argument("--short", action="store_true", help="Use short data")
parser.add_argument("--llm", action="store_true", help="creates llm inputs based on test data")
args = parser.parse_args()


# Get the absolute path to the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LIST_TO_DROP = open("list_to_drop.txt", "r").read().splitlines()


# Add the root dir to sys.path so Python can find the 'rsd' package
sys.path.insert(0, ROOT_DIR)

LLM_LABELS_MAP = {
    0.0: 5,
    0.2: 4,
    0.4: 3,
    0.6: 2,
    0.8: 1,
    1.0: 0,
    -1.0: -1,
}

RSD_LABELS_MAP = {
    5: 1,
    4: 0.8,
    3: 0.6, 
    2: 0.4,
    1: 0.2,
    0: 0.0,
    -1: -1.0,
}

if args.lang:
    prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / f"template_{args.lang}_admin.txt"
else:
    prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / "template.txt"
prompt_template = jinja2.Template(prompt_template_path.read_text())

with open("annotations/aligned_pages.json", "r") as f:
    ALIGNED_PAGES = json.load(f)

data_dir = Path(__file__).parent.parent / "data"

if args.lang:
    test_gold_path = data_dir / "evaluation" / "gold_labels" / "full" / f"gold_admin_{args.lang}{'_short' if args.short else ''}.jsonl"
    test_llm_path = data_dir / "evaluation" / "llm_inputs" / "full" / f"test_admin_{args.lang}.jsonl"
else:
    print("Please specify a language with --lang")
    exit()

# Store all text pairs
text_pairs = []
text_pairs_short = []
for i, (page, pages) in enumerate(ALIGNED_PAGES.items()):

    en_text = None
    other_texts = {}
    other_pages = {}    

    text_position_en = None
    text_position_other = None

    # Find English text and other language texts
    for p in pages + [page]:

        if "_en_" in p:
            lang = "en"
            filepath = os.path.join("annotations/cut_parsed_pages/en", p.replace(".html", ".txt"))
            with open(filepath, "r", encoding="utf-8") as f:
                en_text = f.readlines()
                en_text = [line.replace(', ', ' , ').replace('. ', ' . ').replace(';', ' ; ').replace(':', ' : ').split() for line in en_text]
        elif "_de_" in p:
            lang = "de"
            filepath = os.path.join("annotations/cut_parsed_pages/de", p.replace(".html", ".txt")) 
            with open(filepath, "r", encoding="utf-8") as f:
                de_text = f.readlines()
                other_texts["de"] = [line.replace(', ', ' , ').replace('. ', ' . ').replace(';', ' ; ').replace(':', ' : ').split() for line in de_text]
                other_pages["de"] = p
        elif "_fr_" in p:
            lang = "fr"
            filepath = os.path.join("annotations/cut_parsed_pages/fr", p.replace(".html", ".txt"))
            with open(filepath, "r", encoding="utf-8") as f:
                fr_text = f.readlines()
                other_texts["fr"] = [line.replace(', ', ' , ').replace('. ', ' . ').replace(';', ' ; ').replace(':', ' : ').split() for line in fr_text]
                other_pages["fr"] = p
        elif "_it_" in p:
            lang = "it"
            filepath = os.path.join("annotations/cut_parsed_pages/it", p.replace(".html", ".txt"))
            with open(filepath, "r", encoding="utf-8") as f:
                it_text = f.readlines()
                other_texts["it"] = [line.replace(', ', ' , ').replace('. ', ' . ').replace(';', ' ; ').replace(':', ' : ').split() for line in it_text]
                other_pages["it"] = p
    
    #with open(f"annotations/annotations_final/{args.lang}/{page.replace('https___www.', '').replace('.html', '.jsonl')}", "r", encoding="utf-8") as f:
    # uncomment again once sanity check for reviewer is done
    #try:
    with open(f"annotations/annotations_final/{args.lang}/{page.replace('https___www.', '').replace('.html', '.jsonl')}", "r", encoding="utf-8") as f:
        
        if page.replace('https___www.', '').replace('.html', '.jsonl') in os.listdir(f"annotations/annotations_raw/annotations2/{args.lang}"):
            annotator_tag = 2
        else:
            annotator_tag = 1
        # text1 = english   
        # text2 = french
        #print(page)
        if args.lang == "fr" or args.lang == "it" or args.lang == "de":

            annotations = [json.loads(line) for line in f]
            labels_en = [[0] * len(line) for line in en_text]
            labels_other = [[0] * len(line) for line in other_texts[args.lang]]

            for annotation in annotations:
                for text_position in annotation["text2_positions"]:
                    token = en_text[text_position[0]][text_position[1]]
                    assert token in annotation["text2"]
                    # Set score to -1 for punctuation, otherwise use annotation score
                    if token in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                        labels_en[text_position[0]][text_position[1]] = -1
                    else:
                        labels_en[text_position[0]][text_position[1]] = annotation['score']
            
                for text_position in annotation["text1_positions"]:
                    token = other_texts[args.lang][text_position[0]][text_position[1]]
                    assert token in annotation["text1"]
                    # Set score to -1 for punctuation, otherwise use annotation score 
                    if token in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                        labels_other[text_position[0]][text_position[1]] = -1
                    else:
                        labels_other[text_position[0]][text_position[1]] = annotation['score']

            # Get final tokens and labels for both languages
            en_tokens = [token for line in en_text for token in line]
            en_final_labels = [RSD_LABELS_MAP[label] for line in labels_en for label in line]
            
            other_tokens = [token for line in other_texts[args.lang] for token in line]
            other_final_labels = [RSD_LABELS_MAP[label] for line in labels_other for label in line]

            # Process English tokens and labels
            processed_en_tokens = []
            processed_en_labels = []
            
            for token, label in zip(en_tokens, en_final_labels):
                # Check if token has attached punctuation
                if any(p in token for p in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]):
                    # Split token into parts
                    parts = []
                    current_part = ""
                    for char in token:
                        if char in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                            if current_part:
                                parts.append(current_part)
                            parts.append(char)
                            current_part = ""
                        else:
                            current_part += char
                    if current_part:
                        parts.append(current_part)
                        
                    # Add tokens and labels
                    for part in parts:
                        processed_en_tokens.append(part)
                        if part in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                            processed_en_labels.append(-1.0)
                        else:
                            processed_en_labels.append(label)
                else:
                    processed_en_tokens.append(token)
                    processed_en_labels.append(label)

            # Process other language tokens and labels  
            processed_other_tokens = []
            processed_other_labels = []
            
            for token, label in zip(other_tokens, other_final_labels):
                # Check if token has attached punctuation
                if any(p in token for p in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]):
                    # Split token into parts
                    parts = []
                    current_part = ""
                    for char in token:
                        if char in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                            if current_part:
                                parts.append(current_part)
                            parts.append(char)
                            current_part = ""
                        else:
                            current_part += char
                    if current_part:
                        parts.append(current_part)
                        
                    # Add tokens and labels
                    for part in parts:
                        processed_other_tokens.append(part)
                        if part in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]:
                            processed_other_labels.append(-1.0)
                        else:
                            processed_other_labels.append(label)
                else:
                    processed_other_tokens.append(token)
                    processed_other_labels.append(label)

            assert len(processed_en_tokens) == len(processed_en_labels)
            assert len(processed_other_tokens) == len(processed_other_labels)

            if not args.short:
                text_pairs.append({
                "text_a": " ".join(processed_en_tokens),
                "text_b": " ".join(processed_other_tokens), 
                "labels_a": processed_en_labels,
                "labels_b": processed_other_labels,
                "page_en": page,
                "page_other": other_pages[args.lang],
                "chunk_id": 0,  # Full document is chunk 0
                "id": f"admin_{args.lang}_{i}",
                "annotator_tag": annotator_tag,
            })
            else:
                # Split data into chunks of max 250 tokens
                max_tokens = 250
                chunks = []
                
                # Calculate total tokens for both sentences combined
                total_tokens = len(processed_en_tokens) + len(processed_other_tokens)
                
                if total_tokens <= max_tokens:
                    # If total tokens is within limit, add as single chunk
                    chunks.append({
                        "text_a": " ".join(processed_en_tokens) if not args.llm else processed_en_tokens,
                        "text_b": " ".join(processed_other_tokens) if not args.llm else processed_other_tokens,
                        "labels_a": processed_en_labels,
                        "labels_b": processed_other_labels,
                        "page_en": page,
                        "page_other": other_pages[args.lang],
                        "chunk_id": 0,
                        "id": f"admin_{args.lang}_{i}",
                        "annotator_tag": annotator_tag,
                    }) 
                else:
                    # Split into multiple chunks while trying to keep sentences together
                    start_a = 0
                    start_b = 0
                    chunk_counter = 0
                    
                    while start_a < len(processed_en_tokens) or start_b < len(processed_other_tokens):
                        # Calculate how many tokens we can take from each sentence
                        remaining_tokens = max_tokens
                        end_a = min(start_a + remaining_tokens//2, len(processed_en_tokens))
                        remaining_tokens -= (end_a - start_a)
                        end_b = min(start_b + remaining_tokens, len(processed_other_tokens))
                        
                        # assert that labels have the same length as the tokens
                        assert len(processed_en_tokens[start_a:end_a]) == len(processed_en_labels[start_a:end_a])
                        assert len(processed_other_tokens[start_b:end_b]) == len(processed_other_labels[start_b:end_b])
                        
                        chunks.append({
                            "text_a": " ".join(processed_en_tokens[start_a:end_a]),
                            "text_b": " ".join(processed_other_tokens[start_b:end_b]),
                            "labels_a": processed_en_labels[start_a:end_a],
                            "labels_b": processed_other_labels[start_b:end_b],
                            "page_en": page,
                            "page_other": other_pages[args.lang],
                            "chunk_id": chunk_counter,
                            "id": f"admin_{args.lang}_{i}",
                            "annotator_tag": annotator_tag,
                        })
                        
                        start_a = end_a
                        start_b = end_b
                        chunk_counter += 1
                
                # Add all chunks to text_pairs
                text_pairs.extend(chunks)
    """except FileNotFoundError:
            print(f"File not found: {f}")
            continue"""


with jsonlines.open(test_gold_path, "w") as f:
    doc_counter = -1
    current_page = None
    for text_pair in text_pairs:

        if text_pair["id"] in LIST_TO_DROP:
            continue

        text_pair["subset"] = f"admin_{args.lang}"
        
        # Check if we're moving to a new page BEFORE assigning the ID
        if current_page != text_pair["page_en"]:
            current_page = text_pair["page_en"]
            doc_counter += 1
        
        # This is a chunk, use document ID + chunk ID
        text_pair["id"] = f"admin_{args.lang}_{doc_counter}"
        text_pair["chunk_id"] = text_pair["chunk_id"]  # Keep the chunk_id as is
        
        f.write(text_pair)

    print(f"Wrote {len(text_pairs)} text pairs to {test_gold_path}")

if args.llm:
    doc_counter = 0
    current_page = None
    with jsonlines.open(test_llm_path, "w") as f:
        for text_pair in text_pairs:

            if text_pair["id"] in LIST_TO_DROP:
                continue

            prompt = prompt_template.render(
                sentence1=json.dumps(text_pair["text_a"]),
                sentence2=json.dumps(text_pair["text_b"]),
            )
            gold_response = {
                    "sentence1": [[token, LLM_LABELS_MAP[round(label, 1)]] for token, label in zip(text_pair["text_a"], text_pair["labels_a"])],
                    "sentence2": [[token, LLM_LABELS_MAP[round(label, 1)]] for token, label in zip(text_pair["text_b"], text_pair["labels_b"])],
                }
            f.write({
                "messages": [{"role": "user", "content": prompt}],
                "id": f"admin_{args.lang}_{doc_counter}",  # Use the same ID that was generated for the gold dataset
            })

            # Check if we're moving to a new page
            if current_page != text_pair["page_en"]:
                current_page = text_pair["page_en"]
                doc_counter += 1
        print(f"Wrote {len(text_pairs)} text pairs to {test_llm_path}")


