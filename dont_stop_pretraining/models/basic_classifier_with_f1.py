from typing import Dict, Optional, List, Union

from overrides import overrides
import torch
import numpy as np


from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("basic_classifier_with_f1")
class BasicClassifierWithF1(Model):
    """
    This ``Model`` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a ``Seq2SeqEncoder``. The
    resulting sequence is pooled using a ``Seq2VecEncoder`` and then passed to
    a linear classification layer, which projects into the label space. If a
    ``Seq2SeqEncoder`` is not provided, we will pass the embedded text directly to the
    ``Seq2VecEncoder``.

    This model additionally provides F1 measure for classification.
    
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the input text into a ``TextField``
    seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : ``Seq2VecEncoder``
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    dropout : ``float``, optional (default = ``None``)
        Dropout percentage to use.
    num_labels: ``int``, optional (default = ``None``)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace: ``str``, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward_layer: FeedForward,
        seq2seq_encoder: Seq2SeqEncoder = None,
        dropout: Union[str, float] = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        track_weights: bool = False,
        disable_layers: List[str] = []
    ) -> None:

        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._track_weights = track_weights


        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder

        if dropout:
            self._dropout = torch.nn.Dropout(float(dropout))
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._feedforward_layer = feedforward_layer
        self._classifier_input_dim = self._feedforward_layer.get_output_dim()

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._label_f1_metrics: Dict[str, F1Measure] = {}
        for i in range(self._num_labels):
            self._label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)
        self._loss = torch.nn.CrossEntropyLoss()
        self._initial_params = {}
        initializer(self)
        for name, param in self.named_parameters():
            self._initial_params[name] = param.detach().clone()

        if "ff" in disable_layers:
            for name, param in self.named_parameters():
                if "intermediate.dense" in name or "output.dense" in name and "attention" not in name:
                    param.requires_grad = False

        if "attention" in disable_layers:
            for name, param in self.named_parameters():
                if "attention" in name:
                    param.requires_grad = False

        if "layer_norm" in disable_layers:
            for name, param in self.named_parameters():
                if "LayerNorm" in name:
                    param.requires_grad = False

        if "embedding" in disable_layers:
            for name, param in self.named_parameters():
                if "embedding" in name:
                    param.requires_grad = False

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        feedforward_output = self._feedforward_layer(embedded_text)

        logits = self._classification_layer(feedforward_output)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            output_dict['probs'] = probs
            for i in range(self._num_labels):
                metric = self._label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(probs, label)
            self._accuracy(logits, label)

        return output_dict

    def compute_l2_distance(self, X, Y):
        return torch.dist(X.to(Y), Y, 2).item()


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        sum_f1 = 0.0
        for name, metric in self._label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            sum_f1 += metric_val[2]
        if self._track_weights:
            for name, parameter in self.named_parameters():
                metric_dict[name + "_l2_distance"] = self.compute_l2_distance(self._initial_params[name], parameter)
        names = list(self._label_f1_metrics.keys())
        total_len = len(names)
        average_f1 = sum_f1 / total_len
        metric_dict['f1'] = average_f1
        metric_dict['accuracy'] = self._accuracy.get_metric(reset)
        return metric_dict
