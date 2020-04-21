from typing import Tuple, List, Dict
from transformers import  AutoModelWithLMHead, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from tqdm import tqdm, trange
import random
import argparse
import logging
import os
import json


logger = logging.getLogger(__name__)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class LineByLineTextDataset(Dataset):
    def __init__(self, examples):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_batch_size', type=int, help="batch size", default=1)
    parser.add_argument('--block_size', type=int, help="max length", default=512)
    parser.add_argument('--seed', type=int, help="masking seed", default=900)
    parser.add_argument('--mlm_probability', type=float, help='masking probability', default=0.15)
    parser.add_argument('--model_name_or_path', type=str, help='roberta model', required=True)
    parser.add_argument('--input_file', type=str, help='path to data sample', required=True)
    parser.add_argument('--mlm', action='store_true', help='mlm loss or regular loss')
    parser.add_argument('--sample', type=int, help='sample', default=None)
    parser.add_argument('--sampling_seeds', nargs='+', help='seeds', type=int, default=None)

    args = parser.parse_args()

    if args.eval_batch_size > 1:
        raise ValueError("To reproduce the results in paper, the batch size has to be 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading model...')
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_loss = 0.0
    nb_eval_steps = 0
    losses = []
    # with open(args.input_file, encoding="utf-8") as f:
    #     lines = [line for line in tqdm(f.read().splitlines()) if (len(line) > 0 and not line.isspace())]
    
    # examples = tokenizer.batch_encode_plus(tqdm(lines), add_special_tokens=True, max_length=args.block_size)["input_ids"]
    examples = []
    with open(args.input_file) as f:
        for ix, line in enumerate(f):
            try:
                examples.append(json.loads(line.strip())[0])
            except:
                continue
        # examples = [json.loads(line) for line in f if line]
    print(len(examples))
    pbar = tqdm(args.sampling_seeds)
    losses = []
    for seed in pbar:
        eval_loss = 0.0
        nb_eval_steps = 0
        print('Making dataset...')
        np.random.seed(seed)
        examples_sub = np.random.choice(examples, args.sample)
        eval_dataset = LineByLineTextDataset(examples=examples_sub)
        print('Making sampler...')
        eval_sampler = SequentialSampler(eval_dataset)
        print('Making dataloader...')
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
        )
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if args.mlm:
                inputs, labels = mask_tokens(batch, tokenizer, args.mlm_probability)
                inputs = inputs.to(device)
                labels = labels.to(device)
            else:
                inputs = batch.to(device)
                labels = [list(map(lambda x: -100 if x == tokenizer.pad_token_id else x, val)) for val in inputs.tolist()]
                labels = torch.tensor(labels)
                labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        losses.append(eval_loss)
        pbar.set_description(f"{eval_loss}:")
        # print ('masked LM loss for model: %f\n' % eval_loss)
    losses = np.array(losses)
    print(f"loss: {losses.mean()} +- {losses.std()}")
