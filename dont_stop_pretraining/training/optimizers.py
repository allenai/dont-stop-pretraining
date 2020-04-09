

import logging
import math
import re
from typing import Any, Dict, List, Tuple, Union

import torch
import transformers
from allennlp.training.optimizers import Optimizer, make_parameter_groups


@Optimizer.register("huggingface_adamw_str_lr")
class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW):
    """
    Registered as an `Optimizer` with name "huggingface_adamw".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: Union[str, float] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-06,
        weight_decay: float = 0.0,
        correct_bias: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=float(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )



@Optimizer.register("adam_str_lr")
class AdamOptimizer(Optimizer, torch.optim.Adam):
    """
    Registered as an `Optimizer` with name "adam".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: Union[str, float] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=float(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
