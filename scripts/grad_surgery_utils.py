import torch
import numpy as np

DELTA = 1e-10

# My personal implementation of Gram-Schmidt orthogonalization
def my_gramschmidt(tensor):
	for i in range(tensor.shape[0]):
		if i == 0:
			tensor[i].div_(tensor[i].norm() + DELTA)
		else:
			proj_ = (tensor[i].unsqueeze(0)).matmul(tensor[:i].t()).matmul(tensor[:i])
			tensor[i] = tensor[i] - proj_
			tensor[i].div_(tensor[i].norm() + DELTA)
	return tensor

def vectorize(list_of_tensors):
	orig_shapes, vec = [], []
	with torch.no_grad():
		for tensor in list_of_tensors:
			orig_shapes.append(tensor.shape)
			vec.append(tensor.view(-1))
		vec = torch.cat(vec)
	return orig_shapes, vec

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)