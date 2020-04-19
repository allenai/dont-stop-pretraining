import glob
from tqdm import tqdm
import numpy as np
import torch
import sys
from numpy.lib.format import open_memmap
import os

if __name__ == '__main__':
    input_dir = sys.argv[1]
    dirs = glob.glob(input_dir)
    for file_ in tqdm(dirs):
        if ".emb" not in file_ and ".id" not in file_:
            x = torch.load(file_)
            mat = x[1].detach().numpy()
            ids = x[0].detach().numpy()
            fp_mat = open_memmap(file_ + ".emb.npy", dtype=np.float32, mode='w+', shape=(mat.shape[0], mat.shape[1]))
            fp_mat[...] = mat
            fp_mat.flush()
            fp_ids = open_memmap(file_ + ".id.npy", dtype=np.float32, mode='w+', shape=(ids.shape[0], ids.shape[1]))
            fp_ids[...] = ids
            fp_ids.flush()
            os.remove(file_)