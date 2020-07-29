import argparse
import json
import os
import sys
from typing import Iterator

import torch
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np
from vampire.api import VampireModel
from vampire.common.util import load_sparse
import zipfile
from scipy import sparse
import logging


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in tqdm(zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:])):
             data.append(csr_matrix.data[row_start:row_end])
             indices.append(csr_matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]

        return sparse.csr_matrix((data, indices, indptr), shape=shape)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="path to vampire model archive (e.g. /path/to/model.tar.gz)"
                             " or huggingface model name (e.g. roberta-base) ")
    parser.add_argument("--output_file", type=str, required=True, help='path to output')
    parser.add_argument("--input_file", type=str, required=True, help='path to output')
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--device', type=int, required=False, default=-1)
    torch.set_num_threads(64)
    args = parser.parse_args()
    vectors = []
    ids = []

    if 'model.tar.gz' in args.model:
        model = VampireModel.from_pretrained(args.model, args.device, for_prediction=True)
    else:
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.device >= 0:
            model = model.to(f'cuda:{args.device}')

    with np.load(args.input_file, allow_pickle=True) as data:
        mat = data['emb'].item()
        ids_ = data['ids']
    row_indexer = SparseRowIndexer(mat)
    indices = list(range(mat.shape[0]))
    if os.path.exists(args.output_file):
        ids_already_done, _ = torch.load(args.output_file)
    else:
        ids_already_done = []
    indices = list(set(indices) - set(ids_already_done))
    logging.info(f"{len(ids_already_done)} examples already processed.")
    indices_batches = batch(indices, n=args.batch_size)
    batch_embs = []
    batch_ids_ = []
    
    for target_indices in tqdm(indices_batches, total=len(indices) // args.batch_size):
        batch_rows = row_indexer[target_indices].toarray()
        batch_ids = ids_[target_indices]
        batch_embs.append(batch_rows)
        batch_ids_.append(batch_ids)
    
    try:
        for batch_rows, batch_ids in tqdm(zip(batch_embs, batch_ids_), total=len(batch_embs)):
            if 'tar.gz' in args.model:
                batch_vectors = model.extract_features(batch_rows,
                                                    batch=True,
                                                    scalar_mix=True)
                vectors.extend([x.cpu() for x in batch_vectors])
            indices = torch.IntTensor(batch_ids).unsqueeze(-1)
            ids.append(indices)
    except:
        logging.info("Feature extraction failed, saving ids and vectors so far...")
        torch.save((torch.cat(ids, 0).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)
    torch.save((torch.cat(ids, 0).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)