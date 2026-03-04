import os
import json
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt


parser = ap.ArgumentParser()
parser.add_argument("--lang", type=str, default="all")
parser.add_argument("--viz", action="store_true")
parser.add_argument("--save_name", type=str, default="")
args = parser.parse_args()

annotations_dir = "annotations/annotations_final"


def main():
    # put all annotation into a dataframe
    annotations = []

    for lang_dir in os.listdir(annotations_dir):
        if args.lang != "all" and args.lang not in lang_dir or "discarded" in lang_dir:
            continue

        path_to_annotation = os.path.join(annotations_dir, lang_dir)

        for file in os.listdir(path_to_annotation):
            if not file.endswith(".jsonl"):
                continue

            #print(f"Processing {path_to_annotation}/{file}")

            with open(os.path.join(path_to_annotation, file), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ann = json.loads(line)
                        annotations.append(ann)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {file}")
                        continue
    
    df = pd.DataFrame(annotations)

    #print(f"Total annotations loaded: {len(df)}")
    #print("\nFirst few rows:")

    #print(df)

    score1 = df[df["score"] == 1]
    score2 = df[df["score"] == 2]
    score3 = df[df["score"] == 3]
    score4 = df[df["score"] == 4]
    score5 = df[df["score"] == 5]

    print(len(score1))
    print(len(score2))
    print(len(score3))
    print(len(score4))
    print(len(score5))
    
    """for index, row in score1.iterrows():
        print(row["text1"], row["text2"])
        print('\n')"""


    # visualize the distribution of scores in the annotations
    if args.viz:
        plt.figure(figsize=(10, 5))
        plt.hist(df["score"], bins=range(1,7), align='left', rwidth=0.8, edgecolor="black")
        plt.title(f"Distribution of all scores") if args.lang == "all" else plt.title(f"Distribution of scores for {args.lang}")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.xticks(range(1,6), range(1,6))  # Set x-axis tick locations and labels
        plt.show()

    # save the visualization to a file
    plt.savefig(f"annotations/visualizations/{args.save_name}{args.lang}.png")



if __name__ == "__main__":
    main()
