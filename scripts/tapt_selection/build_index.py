import faiss
import torch
import glob
import argparse
from tqdm import tqdm, trange
import simplejson as json
from itertools import islice
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
import os
from tempfile import mkdtemp
import re


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
        for key, val in tqdm(list_of_json_objects.items()):
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
    parser.add_argument('--vecs', required=True, type=str)
    parser.add_argument('--dim', required=True, type=int)
    parser.add_argument('--text', required=True, type=str)
    parser.add_argument('--serialization_dir', required=True, type=str)
    parser.add_argument('--index_type', required=False, default="Flat", type=str)
    parser.add_argument('--batch-size', required=False, default=64, type=int)
    parser.add_argument('--device', required=False, default=-1, type=int)

    args = parser.parse_args()

    logger.info("loading vecs...")

    macro_prefixes = list(set([os.path.join(args.vecs, re.findall("\d+", x)[0]) for x in os.listdir(args.vecs)]))
    macro_processor = Processor(macro_prefixes)

    logger.info("building index...")
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    index = faiss.index_factory(args.dim, args.index_type, faiss.METRIC_INNER_PRODUCT)
    if args.index_type != "Flat":
        macro_mat_sample = macro_processor.sample_across_mmap_shards(suffix='.emb.npy', sample=1000000)
        logger.info(f"training...")
        index.train(macro_mat_sample) # train on a large subset of macro data
    logger.info("index built!")
    logger.info("adding all vectors to index...")
    macro_ids = []
    for mat_batch, id_batch in macro_processor.iterate_across_mmap_shards(batch_size=args.batch_size):
        faiss.normalize_L2(mat_batch)
        index.add(mat_batch)   # add vectors to the index
        macro_ids.append(id_batch)
    macro_ids = np.concatenate(macro_ids, axis=0)

    macro_instances = read_dataset(args.text)

    if args.device >= 0:
        index = faiss.index_cpu_to_gpu(res, args.device, index, co)

    logger.info(f"saving index to {args.serialization_dir}...")
    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    faiss.write_index(faiss.index_gpu_to_cpu(index), os.path.join(args.serialization_dir, "faiss.index"))
    write_dataset(macro_instances,  os.path.join(args.serialization_dir, "text.jsonl"))
    np.save(os.path.join(args.serialization_dir, "ids.npy"), macro_ids)
    logger.info("done!")