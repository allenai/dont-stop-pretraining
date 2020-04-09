# pylint: disable=arguments-differ

import torch
from overrides import overrides
from allennlp.common import Registrable
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import (get_final_encoder_states, masked_max, masked_mean, masked_log_softmax)
from allennlp.common.checks import ConfigurationError

class Encoder(Registrable, torch.nn.Module):
    """
    This module is a wrapper over AllenNLP encoders, to make it easy to switch
    between them in the training config when doing things like hyperparameter search.

    It's the same interface as the AllenNLP encoders, except the encoder architecture is
    nested one-level deep (under the field ``architecture``).
    """
    default_implementation = 'feedforward'

    def __init__(self, architecture: torch.nn.Module) -> None:
        super(Encoder, self).__init__()
        self._architecture = architecture

    def get_output_dim(self) -> int:
        return self._architecture.get_output_dim()

    def forward(self, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

@Encoder.register("feedforward")
class MLP(Encoder):

    def __init__(self, architecture: FeedForward) -> None:
        super(MLP, self).__init__(architecture)
        self._architecture = architecture

    @overrides
    def forward(self, **kwargs) -> torch.FloatTensor:
        return self._architecture(kwargs['embedded_text'])

@Seq2VecEncoder.register("maxpool")
class MaxPoolEncoder(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int) -> None:
        super(MaxPoolEncoder, self).__init__()
        self._embedding_dim = embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  #pylint: disable=arguments-differ
        broadcast_mask = mask.unsqueeze(-1).float()
        one_minus_mask = (1.0 - broadcast_mask).byte()
        replaced = tokens.masked_fill(one_minus_mask, -1e-7)
        max_value, _ = replaced.max(dim=1, keepdim=False)
        return max_value

@Encoder.register("seq2vec")
class Seq2Vec(Encoder):

    def __init__(self, architecture: Seq2VecEncoder) -> None:
        super(Seq2Vec, self).__init__(architecture)
        self._architecture = architecture

    @overrides
    def forward(self, **kwargs) -> torch.FloatTensor:
        return self._architecture(kwargs['embedded_text'], kwargs['mask'])


@Encoder.register("seq2seq")
class Seq2Seq(Encoder):

    def __init__(self, architecture: Seq2SeqEncoder, aggregations: str) -> None:
        super(Seq2Seq, self).__init__(architecture)
        self._architecture = architecture
        self._aggregations = aggregations
        if "attention" in self._aggregations:
            self._attention_layer = torch.nn.Linear(self._architecture.get_output_dim(),
                                                    1)

    @overrides
    def get_output_dim(self):
        return self._architecture.get_output_dim() * len(self._aggregations)

    @overrides
    def forward(self, **kwargs) -> torch.FloatTensor:
        mask = kwargs['mask']
        embedded_text = kwargs['embedded_text']
        encoded_output = self._architecture(embedded_text, mask)
        encoded_repr = []
        for aggregation in self._aggregations:
            if aggregation == "meanpool":
                broadcast_mask = mask.unsqueeze(-1).float()
                context_vectors = encoded_output * broadcast_mask
                encoded_text = masked_mean(context_vectors,
                                           broadcast_mask,
                                           dim=1,
                                           keepdim=False)
            elif aggregation == 'maxpool':
                broadcast_mask = mask.unsqueeze(-1).float()
                context_vectors = encoded_output * broadcast_mask
                encoded_text = masked_max(context_vectors,
                                          broadcast_mask,
                                          dim=1)
            elif aggregation == 'final_state':
                is_bi = self._architecture.is_bidirectional()
                encoded_text = get_final_encoder_states(encoded_output,
                                                        mask,
                                                        is_bi)
            elif aggregation == 'attention':
                alpha = self._attention_layer(encoded_output)
                alpha = masked_log_softmax(alpha, mask.unsqueeze(-1), dim=1).exp()
                encoded_text = alpha * encoded_output
                encoded_text = encoded_text.sum(dim=1)
            else:
                raise ConfigurationError(f"{aggregation} aggregation not available.")
            encoded_repr.append(encoded_text)

        encoded_repr = torch.cat(encoded_repr, 1)
        return encoded_repr
