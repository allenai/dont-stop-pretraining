import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.nn import Activation
from overrides import overrides
from torch import nn
from torch.nn import Conv1d, Linear
import logging
import os

logger = logging.getLogger(__name__)

def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


class GetSubnetHattie(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        sigmoid_mask = torch.sigmoid(scores)
        mask = sigmoid_mask * torch.bernoulli(sigmoid_mask)
        return mask

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class GetSubnetMitchell(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        # Get the supermask by sorting the scores and using the top k%
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))     #Changes

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None



class Supermask(nn.Module):
    def __init__(self, weight, type, sparsity, factorized=False):  # in MLP: n_state=3072 (4 * n_embd)
        super(Supermask, self).__init__()
        self.type = type
        self.zeros = torch.zeros_like(weight)
        self.ones = torch.ones_like(weight)
        self.sparsity = sparsity
        self._factorized = factorized
        if factorized:
            self.weight_1 = nn.Parameter(torch.Tensor(weight.size(0), 512))
            self.weight_2 = nn.Parameter(torch.Tensor(512, weight.size(1)))
            if self.training:
                self.weight_1.requires_grad = True
                self.weight_2.requires_grad = True
            else:
                self.weight_1.requires.grad = False
                self.weight_2.requires.grad = False
            if self.training:
                nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
            if self.type == 'HATTIE':
                self.init_mask = GetSubnetHattie.apply(self.weight)
            elif self.type == "MITCHELL":
                self.init_mask = GetSubnetMitchell.apply(self.weight.abs(), self.zeros, self.ones, self.sparsity)
            elif self.type == "LHUC":
                self.init_mask = 2 * torch.sigmoid(torch.mm(self.weight_1, self.weight_2))
        else:
            self.weight = nn.Parameter(torch.Tensor(weight.size()))
            if self.training:
                self.weight.requires_grad = True
            else:
                self.weight.requires_grad = False
            if self.training:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.type == 'HATTIE':
                self.init_mask = GetSubnetHattie.apply(self.weight)
            elif self.type == "MITCHELL":
                self.init_mask = GetSubnetMitchell.apply(self.weight.abs(), self.zeros, self.ones, self.sparsity)
            elif self.type == "LHUC":
                self.init_mask = 2 * torch.sigmoid(self.weight)

        self.curr_mask = self.init_mask
        self.prev_mask = None


    @classmethod
    def from_pretrained(cls, path, type):
        device_ = torch.device('cpu') if not torch.cuda.is_available() else None
        logger.info(f"adding pretrained supermask from {path}")
        supermask = torch.load(path, map_location=device_)
        sm = cls(weight=supermask, type=type, sparsity=1.0)
        sm.curr_mask = supermask
        sm.init_mask = supermask
        return sm
        
    def forward(self, override=False):
        if self._factorized:
            if self.training and self.weight_1.requires_grad:
                if self.type == 'HATTIE':
                    mask = GetSubnetHattie.apply(self.weight)
                elif self.type == "MITCHELL":
                    mask = GetSubnetMitchell.apply(self.weight.abs(), self.zeros, self.ones, self.sparsity)
                elif self.type == "LHUC":
                    mask = 2 * torch.sigmoid(torch.mm(self.weight_1, self.weight_2))
                self.prev_mask = self.curr_mask
                self.curr_mask = mask
            else:
                mask = self.curr_mask
        else:
            if self.training and self.weight.requires_grad:
                if self.type == 'HATTIE':
                    mask = GetSubnetHattie.apply(self.weight)
                elif self.type == "MITCHELL":
                    mask = GetSubnetMitchell.apply(self.weight.abs(), self.zeros, self.ones, self.sparsity)
                elif self.type == "LHUC":
                    mask = 2 * torch.sigmoid(self.weight)
                self.prev_mask = self.curr_mask
                self.curr_mask = mask
            else:
                mask = self.curr_mask
        return mask



class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supermask = None
    
    def add_supermask(self, type, sparsity=None):
        self.supermask = Supermask(weight=self.weight, type=type, sparsity=sparsity)

    def add_pretrained_supermask(self, path, type):
        self.supermask = Supermask.from_pretrained(path, type)

    def forward(self, x):
        if self.supermask is not None:
            supermask = self.supermask()
            w = self.weight * supermask.to(self.weight)
        else:
            w = self.weight
        return torch.nn.functional.linear(x, w)



class SupermaskFeedForward(torch.nn.Module, FromParams):
    """
    This `Module` is a feed-forward neural network, just a sequence of `Linear` layers with
    activation functions in between.

    # Parameters

    input_dim : `int`, required
        The dimensionality of the input.  We assume the input has shape `(batch_size, input_dim)`.
    num_layers : `int`, required
        The number of `Linear` layers to apply to the input.
    hidden_dims : `Union[int, List[int]]`, required
        The output dimension of each of the `Linear` layers.  If this is a single `int`, we use
        it for all `Linear` layers.  If it is a `List[int]`, `len(hidden_dims)` must be
        `num_layers`.
    activations : `Union[Callable, List[Callable]]`, required
        The activation function to use after each `Linear` layer.  If this is a single function,
        we use it after all `Linear` layers.  If it is a `List[Callable]`,
        `len(activations)` must be `num_layers`.
    dropout : `Union[float, List[float]]`, optional (default = 0.0)
        If given, we will apply this amount of dropout after each layer.  Semantics of `float`
        versus `List[float]` is the same as with other parameters.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Union[Activation, List[Activation]],
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:

        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError(
                "len(hidden_dims) (%d) != num_layers (%d)" % (len(hidden_dims), num_layers)
            )
        if len(activations) != num_layers:
            raise ConfigurationError(
                "len(activations) (%d) != num_layers (%d)" % (len(activations), num_layers)
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                "len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers)
            )
        self._activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(SupermaskLinear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def add_pretrained_supermask(self, path, type):
        for ix, layer in enumerate(self._linear_layers):
            layer.add_pretrained_supermask(os.path.join(path, f'_feedforward_layer._linear_layers.{ix}.supermask'), type)

    def add_supermask(self, type, sparsity=None):
        for layer in self._linear_layers:
            layer.add_supermask(type, sparsity)

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            output = dropout(activation(layer(output)))
        return output



class SupermaskConv1d(nn.Conv1d):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    .. note::

        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\text{in}=C_{in}, C_\text{out}=C_{in} \times K, ..., \text{groups}=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize the scores

    def add_supermask(self, type, sparsity=None):
        self.supermask = Supermask(weight=self.weight, type=type, sparsity=sparsity)

    def add_pretrained_supermask(self, path, type):
        self.supermask = Supermask.from_pretrained(path, type)


    def forward(self, x):
        supermask = self.supermask()
        w = self.weight * supermask
        return nn.functional.conv1d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

@Seq2VecEncoder.register("cnn_with_supermask")
class SupermaskCnnEncoder(Seq2VecEncoder):
    """
    A `CnnEncoder` is a combination of multiple convolution layers and max pooling layers.  As a
    [`Seq2VecEncoder`](./seq2vec_encoder.md), the input to this module is of shape `(batch_size, num_tokens,
    input_dim)`, and the output is of shape `(batch_size, output_dim)`.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filters`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    # Parameters

    embedding_dim : `int`, required
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters : `int`, required
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes : `Tuple[int]`, optional (default=`(2, 3, 4, 5)`)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation : `Activation`, optional (default=`torch.nn.ReLU`)
        Activation to use after the convolution layers.
    output_dim : `Optional[int]`, optional (default=`None`)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is `None`, we will just return the result of the max pooling,
        giving an output of shape `len(ngram_filter_sizes) * num_filters`.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        conv_layer_activation: Activation = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name("relu")()
        self._output_dim = output_dim

        self._convolution_layers = [
            SupermaskConv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size,
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = SupermaskLinear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    def add_supermask(self, type, sparsity):
        for conv_layer in self._convolution_layers:
            conv_layer.add_supermask(type, sparsity)
        if self._output_dim:
            self.projection_layer.add_supermask(type, sparsity)

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = (
            torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        )

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result
