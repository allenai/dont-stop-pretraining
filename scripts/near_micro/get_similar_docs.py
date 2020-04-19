import torch
import sys
import pandas as pd
from tqdm import tqdm

# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(micro, world):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(micro, world.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.abs().sqrt()


if __name__ == '__main__':
    micro = torch.load(sys.argv[1])
    world = torch.load(sys.argv[2])
    micro_index = micro[0]
    micro_mat = micro[1]
    world_index = world[0]
    world_mat = world[1]
    k = int(sys.argv[3])
    micro_text = sys.argv[4]
    world_text = sys.argv[5]
 
    micro_df = pd.read_json(micro_text, lines=True).set_index('index')
    world_df = pd.read_json(world_text, lines=True).set_index('index')
    
    print("computing similarity...")
    sim = similarity_matrix(micro_mat, world_mat)
    near_micro = torch.topk(sim, k, dim=1, largest=False)[1]
    print("done!")
    res = pd.DataFrame()
    micro_index = micro_index.squeeze(1).tolist()
    near_micro = near_micro[:, 1:].squeeze(1).tolist()
    for micro_ind, neighbors in tqdm(zip(micro_index, near_micro), total=near_micro.shape[0]):
        sub = {}
        sub['micro'] = [micro_df.loc[micro_ind].text]
        neighbors = [world_index[neighbor].data.numpy()[0] for neighbor in neighbors]
        for ix, neighbor in enumerate(neighbors):
            sub[f'neighbor_{ix}'] = [world_df.loc[neighbor].text]
        res = pd.concat([res, pd.DataFrame(sub)], 0)
    res = res.reset_index(drop=True)
    import ipdb; ipdb.set_trace()