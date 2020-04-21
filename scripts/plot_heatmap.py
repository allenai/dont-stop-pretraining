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

sns.set(context="paper", style="white", font_scale=1.9, font="Times New Roman") 

if __name__ == '__main__':
    
    with open("overlaps_without_stopwords", "r") as f:
        overlaps = json.load(f)
    data = []
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
    plt.tight_layout()
    plt.savefig("heatmap.pdf", dpi=300)