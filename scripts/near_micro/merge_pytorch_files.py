import glob
from tqdm import tqdm
import torch
import sys
import os

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    dirs = glob.glob(input_dir)
    vecs = []
    ids = []
    for file_ in tqdm(dirs):
        x = torch.load(file_)
        vecs.append(x[1])
        ids.append(x[0])
    torch.save((torch.cat(ids,0), torch.cat(vecs, 0)), os.path.join("./micro_emb_swabha", output_file))
