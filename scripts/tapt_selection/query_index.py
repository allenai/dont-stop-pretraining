import faiss
import torch
import glob
import argparse
from tqdm import tqdm, trange
import json
from itertools import islice
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
import os
from tempfile import mkdtemp
import re
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, glob, shard_dim, vec_dim):
        self.mats = []
        for file in glob:
            file_ = np.memmap(file, dtype='float32', mode='r', shape=(shard_dim, vec_dim))
            self.mats.append(file_)
        
    def iterate_efficiently(input, output, chunk_size):
        # create an empty array to hold each chunk
        # the size of this array will determine the amount of RAM usage
        holder = np.zeros([chunk_size,800,800], dtype='uint16')

        # iterate through the input, replace with ones, and write to output
        for i in range(input.shape[0]):
            if i % chunk_size == 0:
                holder[:] = input[i:i+chunk_size] # read in chunk from input
                holder += 5 # perform some operation
                output[i:i+chunk_size] = holder # write chunk to output


    def __len__(self):
        return sum([len(x) for x in self.mats])

    def __getitem__(self, idx):
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def write_dataset(list_of_json_objects, output_filename):
    with open(output_filename, 'w') as output_file:
        for key, val in list_of_json_objects.items():
            output_file.write(json.dumps({"index": key, "text": val}) + '\n')


def read_dataset(file_path:str):
  """
  Reads jsonl file to recover mapping between IDs and content of remaining data.
  """
  instances = {}
  with open(file_path, 'r') as file_:
    for line in tqdm(file_):
      example = json.loads(line)
      assert "index" in example
      instances[example["index"]] = example["text"]

  return instances

def batchify(mat, batch_size):
    batches = np.array_split(mat, int(mat.shape[0] // batch_size))
    for batch in tqdm(batches):
        yield batch

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def fvecs_mmap(fname):
    return np.load(fname)

def sample_across_mmap_shards(sharded_mat, sample):
    res = []
    for submat in sharded_mat:
        res.append(submat[np.random.choice(submat.shape[0], sample // len(sharded_mat), replace=False), :])
    return np.concatenate(res)


class Processor:
    def __init__(self, prefixes):
        self.prefixes = prefixes
        self.num_shards = len(prefixes)

    def sample_across_mmap_shards(self, suffix, sample):
        res = []
        logger.info("sampling...")
        for prefix in tqdm(self.prefixes):
            submat = fvecs_mmap(prefix + suffix)
            res.append(submat[np.random.choice(submat.shape[0], sample // self.num_shards, replace=False), :])
        return np.concatenate(res)

    def iterate_across_mmap_shards(self, batch_size=None, sample=None):
        for prefix in tqdm(self.prefixes):
            submat = fvecs_mmap(prefix + ".emb.npy")
            subids = fvecs_mmap(prefix + ".id.npy")
            if sample:
                idx = np.random.choice(submat.shape[0], sample // self.num_shards, replace=False)
                submat = submat[idx, :]
                subids = subids[idx, :]
            if batch_size:
                mat_batches = np.array_split(submat, int(submat.shape[0] // batch_size))
                id_batches = np.array_split(subids, int(subids.shape[0] // batch_size))
                for mat_batch, id_batch in zip(mat_batches, id_batches):
                    yield mat_batch, id_batch
            else:
                yield submat, subids


    def collapse_mmap_shards(self, suffix):
        batches = []
        for prefix in self.prefixes:
            submat = fvecs_mmap(prefix + suffix)
            batches.append(submat)
        return np.concatenate(batches, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vecs', required=False, type=str)
    parser.add_argument('--dim', required=True, type=int)
    parser.add_argument('--text', required=False, type=str)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('--load-index', required=False, type=str)
    parser.add_argument('--output-file', required=False, type=str)
    parser.add_argument('--batch-size', required=False, default=64, type=int)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--output_neighbor', required=False, type=int)
    parser.add_argument('--device', required=False, default=-1, type=int)
    parser.add_argument('--df', required=False, type=str)
    parser.add_argument('--reorder_macro', required=False, type=str)

    args = parser.parse_args()
    

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    logger.info(f"loading index at {args.load_index}...")
    index = faiss.read_index(os.path.join(args.load_index, "faiss.index"))
    macro_instances = read_dataset(os.path.join(args.load_index, "text.jsonl"))
    macro_ids = np.load(os.path.join(args.load_index, "ids.npy"))

    logger.info(f"index loaded!")

    if args.device >= 0:
        index = faiss.index_cpu_to_gpu(res, args.device, index, co)

    neighbors_ = []
    dists_ = []
    micro_ids = []
    micro_prefixes = list(set([os.path.join(args.vecs, re.findall("\d+", x)[0]) for x in os.listdir(args.vecs)]))
    micro_processor = Processor(micro_prefixes)

    logger.info('searching for nearest neighbors...')
    for mat_batch, id_batch in micro_processor.iterate_across_mmap_shards(batch_size=args.batch_size):
        faiss.normalize_L2(mat_batch)
        dists, ns = index.search(mat_batch, args.k)
        neighbors_.append(ns)
        dists_.append(dists)
        micro_ids.append(id_batch)
    micro_ids = np.concatenate(micro_ids, axis=0)
    neighbors = np.concatenate(neighbors_, axis=0)
    dists = np.concatenate(dists_, axis=0)

    micro_ids = torch.IntTensor(micro_ids.squeeze(-1))
    macro_ids = torch.IntTensor(macro_ids.squeeze(-1))
    
    micro_instances = read_dataset(args.text)
    if args.inspect:
        for i in range(neighbors.shape[0]):
            me = micro_ids[i].item()
            print(f"Source: {micro_instances.get(me)}")
            knn = neighbors[i, :]
            knn_ids = torch.index_select(macro_ids, index=torch.IntTensor(knn).long(), dim=0).tolist()
            for j, kid in enumerate(knn_ids):
                print(f"Neighbor_{j}: {macro_instances.get(kid)}")
            print()

    if args.reorder_macro:
        texts = []

        for i in trange(neighbors.shape[0]): 
            knn = neighbors[i, :]
            knn_ids = torch.index_select(macro_ids, index=torch.IntTensor(knn).long(), dim=0).tolist()
            for j, kid in enumerate(knn_ids):
                text = macro_instances.get(kid)
                subdf = {"text": text, "knn_id": j, 'macro_index': kid}
                texts.append(subdf)
        texts = pd.DataFrame(texts)
        print('reordering...')
        texts = texts.sort_values('knn_id').groupby("text", as_index=False).first().sort_values('knn_id').reset_index(drop=True)
        print('adding extras...')
        extra_ids = set(macro_instances.keys()) - set(texts.macro_index)
        max_knn_id = texts.knn_id.max()
        extras = [{"text": macro_instances[x], "knn_id": max_knn_id + 1, "macro_index": x} for x in tqdm(extra_ids)]        
        texts = pd.concat([texts, pd.DataFrame(extras)])
        for name, group in tqdm(texts.groupby('knn_id')):
            name = str(name)
            if len(name) < 4:
                name = (4 - len(name)) * "0" + name
            with open(os.path.join(args.reorder_macro, f"knn_{name}.txt"), "w+") as f:
                for item in group.text:
                    f.write(item + "\n\n")
    if args.df:
        vals = []
        for i in range(neighbors.shape[0]):
            val = {}
            me = micro_ids[i].item()
            val['source'] = micro_instances.get(me)
            knn = neighbors[i, :]
            kdists = dists[i, :]
            knn_ids = torch.index_select(macro_ids, index=torch.IntTensor(knn).long(), dim=0).tolist()
            for j, kid in enumerate(knn_ids):
                if macro_instances.get(kid) is None:
                    import ipdb; ipdb.set_trace()
                val[f'neighbor_{j}'] = macro_instances.get(kid)
                val[f'dist_{j}'] = kdists[j]
            vals.append(val)
        df = pd.DataFrame(vals)
        df.to_json(args.df, lines=True, orient="records")


    if args.output_file:
        texts = []
        for i in range(neighbors.shape[0]): 
            knn = neighbors[i, :]
            knn_ids = torch.index_select(macro_ids, index=torch.IntTensor(knn).long(), dim=0).tolist()
            for j, kid in enumerate(knn_ids):
                text = macro_instances.get(kid)
                if args.output_neighbor is not None:
                    if j == args.output_neighbor:
                        texts.append(text)
                else:
                    texts.append(text)
        texts = list(set(texts))
        logger.info(f"found {len(texts)} nearest neighbor examples.")
        logger.info(f"writing nearest neighbor examples to {args.output_file}.")
        with open(args.output_file, 'w+') as f:
            for text in tqdm(texts):
                json_ = {"text": text}
                f.write(json.dumps(json_) + "\n")
