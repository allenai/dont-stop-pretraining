import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
from typing import List
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse


sns.set(context="paper", style="white", font_scale=2.1, font="Times New Roman") 

def load_data(data_path: str) -> List[str]:
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {"text": line}
                text = example['text']
                examples.append(text)
    return examples

def load_vocab(file):
    text = load_data(file)
    count_vectorizer = CountVectorizer(min_df=3, stop_words="english")
    count_vectorizer.fit(tqdm(text))
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="files containing tokenized text from each domain", required=True)
    parser.add_argument("--output_file", help="path to save heatmap", required=True)
    args = parser.parse_args()
    files = args.files
    vocabs = {}
    for file in files:
        if 'med' in file:
            key = 'BioMed'
        elif 'review' in file:
            key = 'Reviews'
        elif 'cs' in file:
            key = 'CS'
        elif 'news' in file:
            key = 'News'
        elif 'pretraining' in file:
            key = 'PT'
        vocabs[key] = load_vocab(file)
    
    file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    
    overlaps = {}
    for x, y in file_pairs:
        intersection = vocabs[x] & vocabs[y]
        union = (vocabs[x] | vocabs[y])
        overlaps[x + "_" + y] = len(intersection) / len(union)
    
    data = []

    with open("overlaps_without_stopwords", "w+") as f:
        json.dump(overlaps, f)
z = {}
for key in overlaps.keys():
    file_1, file_2 = key.split('_')
    if not z.get(file_1):
        z[file_1] = {}
    z[file_1][file_2] = overlaps[key]
    if not z.get(file_2):
        z[file_2] = {}
    z[file_2][file_1] = overlaps[key]

labels = ["PT", "News", "Reviews", "BioMed", "CS"]

for ix, key in enumerate(labels):
    items = []
    for subkey in labels:
        if not z[key].get(subkey):
            items.append(1.0)
        else:
            items.append(z[key][subkey])
    data.append(items)
data = np.array(data) * 100
ax = sns.heatmap(data, cmap="Blues", vmin=30, xticklabels=labels, annot=True, fmt=".1f", cbar=False, yticklabels=labels)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(args.output_file, dpi=300)