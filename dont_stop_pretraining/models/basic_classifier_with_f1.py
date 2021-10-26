from typing import Dict, Optional

from overrides import overrides
import torch


from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, MeanAbsoluteError
import pdb

EPS = 1e-8

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
			dropout: float = None,
			num_labels: int = None,
			label_namespace: str = "labels",
			initializer: InitializerApplicator = InitializerApplicator(),
			regularizer: Optional[RegularizerApplicator] = None,
	) -> None:

		super().__init__(vocab, regularizer)
		self._text_field_embedder = text_field_embedder

		if seq2seq_encoder:
			self._seq2seq_encoder = seq2seq_encoder
		else:
			self._seq2seq_encoder = None
	
		self._seq2vec_encoder = seq2vec_encoder
		self._classifier_input_dim = None
		if seq2vec_encoder:
			self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()


		if dropout:
			self._dropout = torch.nn.Dropout(dropout)
		else:
			self._dropout = None

		self._label_namespace = label_namespace
		if num_labels is not None:
			self._num_labels = num_labels
		else:
			self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
		self._feedforward_layer = feedforward_layer
		if self._classifier_input_dim is not None:
			self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
		
		if self._num_labels > 1: # We are performing classification
			self._accuracy = CategoricalAccuracy()
			self._label_f1_metrics: Dict[str, F1Measure] = {}
			for i in range(self._num_labels):
				self._label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)
			self._loss = torch.nn.CrossEntropyLoss(reduction='none')
		if initializer is not None:
			initializer(self)

	def forward(self, tokens, label=None, embedded_text=None, attn_mask=None) -> Dict[str, torch.Tensor]:

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
		if embedded_text is None:
			embedded_text = self._text_field_embedder(tokens, attention_mask=attn_mask)
		if isinstance(tokens, dict):
			mask = get_text_field_mask(tokens).float()
		else:
			mask = None
			embedded_text = embedded_text[1][-1]

		if self._seq2seq_encoder:
			embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

		embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

		if self._dropout:
			embedded_text = self._dropout(embedded_text)

		feedforward_output = self._feedforward_layer(embedded_text)

		logits = self._classification_layer(feedforward_output)

		# We are doing classification
		probs = torch.nn.functional.softmax(logits, dim=-1)
		output_dict = {"logits": logits, "probs": probs}

		if label is not None:
			loss = self._loss(logits, label.long().view(-1))
			output_dict["loss_full"] = loss
			output_dict["loss"] = loss.mean()
			for i in range(self._num_labels):
				metric = self._label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
				metric(probs, label)
			self._accuracy(logits, label)

		return output_dict

	@overrides
	def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		"""
		Does a simple argmax over the probabilities, converts index to string label, and
		add ``"label"`` key to the dictionary with the result.
		"""
		predictions = output_dict["probs"]
		if predictions.dim() == 2:
			predictions_list = [predictions[i] for i in range(predictions.shape[0])]
		else:
			predictions_list = [predictions]
		classes = []
		for prediction in predictions_list:
			label_idx = prediction.argmax(dim=-1).item()
			label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
					label_idx, str(label_idx)
			)
			classes.append(label_str)
		output_dict["label"] = classes
		return output_dict

	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		metric_dict = {}
		sum_f1 = 0.0
		for name, metric in self._label_f1_metrics.items():
			metric_val = metric.get_metric(reset)
			sum_f1 += metric_val[2]
		names = list(self._label_f1_metrics.keys())
		total_len = len(names)
		average_f1 = sum_f1 / total_len
		metric_dict['f1'] = average_f1
		metric_dict['accuracy'] = self._accuracy.get_metric(reset)
		return metric_dict


@Model.register("basic_sequence_classifier_with_f1")
class BasicSequenceTagger(BasicClassifierWithF1):
	def __init__(
			self,
			vocab: Vocabulary,
			text_field_embedder: TextFieldEmbedder,
			feedforward_layer: FeedForward,
			input_dim: int = 768,
			dropout: float = None,
			num_labels: int = None,
			label_namespace: str = "labels",
			max_loss_scale=50,
			initializer: InitializerApplicator = InitializerApplicator(),
			regularizer: Optional[RegularizerApplicator] = None,
	) -> None:

		super().__init__(vocab, text_field_embedder, None, feedforward_layer, None, dropout, num_labels, label_namespace, initializer, regularizer)

		self._classifier_input_dim =  input_dim
		self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
		if self._num_labels == 1: # We are performing regression
			self._mae = MeanAbsoluteError()
			self._loss = torch.nn.MSELoss(reduction='none')
		self.is_classification_task = self._num_labels > 1
		self.max_loss_scale = max_loss_scale
		if initializer is not None:
			initializer(self)

	def forward(  # type: ignore
				self, tokens, label=None, attn_mask=None, embedded_text=None
		) -> Dict[str, torch.Tensor]:

		if embedded_text is None:
			embedded_text = self._text_field_embedder(tokens, attention_mask=attn_mask)
		if isinstance(tokens, dict):
			mask = get_text_field_mask(tokens).float()
		else:
			mask = None
			embedded_text = embedded_text[1][-1]

		if self._dropout:
			embedded_text = self._dropout(embedded_text)

		feedforward_output = self._feedforward_layer(embedded_text)

		logits = self._classification_layer(feedforward_output)

		output_dict = {}
		if self.is_classification_task:
			probs = torch.nn.functional.softmax(logits, dim=-1)
			output_dict = {"logits": logits, "probs": probs}
			label = label.long()

		if label is not None:
			# TODO[LDERY] need to check if we consider the padding appropriately here
			logits, label = logits.view(-1), label.view(-1)
			if not self.is_classification_task:
				label_mask = (label >= 0).float()
				label = label * label_mask
				logits = logits * label_mask
			loss = self._loss(logits, label)
			if not self.is_classification_task:
				scale_mask = (loss > self.max_loss_scale)*1.0
				with torch.no_grad():
					inv_ = 1.0 / ((scale_mask * loss) + (1.0 - scale_mask))
					scale = (scale_mask * self.max_loss_scale) + (1.0 - scale_mask)
					factor = (inv_ * scale).detach()
				loss = loss * factor
			# Todo - ldery - need to bound
			output_dict["loss_full"] = loss
			if not self.is_classification_task:
				output_dict["loss"] = loss.sum() / (label_mask.float().sum())
			else:
				output_dict["loss"] = loss.mean()
			if self.is_classification_task:
				for i in range(self._num_labels):
					metric = self._label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
					metric(probs, label)
				self._accuracy(logits, label)
			else:
				self._mae(logits[label >= 0].view(-1), label[label >= 0].view(-1))
		return output_dict
	
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		if self.is_classification_task:
			return super().get_metrics(reset)
		metric_dict = {}
		metric_dict['mae'] = self._mae.get_metric(reset)
		return metric_dict

'''
Code borrowed from :
https://github.com/StephAO/olfmlm/blob/6be204cdb78cda71b3bf37dd5c0a186e990a55c1/model/new_models.py
'''
def batch_cos_sim(a, b):
	with torch.no_grad():
		norm_a = a.norm(dim=-1)[:, :, None] + EPS
		norm_b = b.norm(dim=-1)[:, None] + EPS
	a_norm = a / norm_a
	b_norm = b / norm_b
	return a_norm.matmul(b_norm.T)


@Model.register("basic_sentence_classifier_with_f1")
class BasicSentenceClassifier(BasicClassifierWithF1):
	def __init__(
			self,
			text_field_embedder: TextFieldEmbedder,
			seq2vec_encoder: Seq2VecEncoder,
			sent_feedforward: FeedForward,
			tok_feedforward: FeedForward,
			input_dim: int = 768,
			dropout: float = None,
			initializer: InitializerApplicator = InitializerApplicator(),
			regularizer: Optional[RegularizerApplicator] = None,
	) -> None:

		super().__init__(None, text_field_embedder, None, None, None, dropout, num_labels=0, label_namespace=None, initializer=initializer, regularizer=regularizer)
		self.sent_feedforward = sent_feedforward
		self.sent_layernorm = torch.nn.LayerNorm(input_dim)
		self.tok_feedforward = tok_feedforward
		self.tok_layernorm = torch.nn.LayerNorm(input_dim)

		self._seq2vec_encoder = seq2vec_encoder
		self._accuracy = CategoricalAccuracy()
		self._loss = torch.nn.CrossEntropyLoss(reduction='none')
		if initializer is not None:
			initializer(self)

	def forward(  # type: ignore
				self, tokens, label=None, attn_mask=None, embedded_text=None
		) -> Dict[str, torch.Tensor]:

		if embedded_text is None:
			embedded_text = self._text_field_embedder(tokens, attention_mask=attn_mask)
		if isinstance(tokens, dict):
			mask = get_text_field_mask(tokens).float()
		else:
			mask = None

		pooled_text = embedded_text[1][-1]
		pooled_text = self._seq2vec_encoder(pooled_text, mask=mask)
		if self._dropout:
			pooled_text = self._dropout(pooled_text)
		pooled_text = self.sent_feedforward(pooled_text)
		pooled_text = self.sent_layernorm(pooled_text)
		if self._dropout:
			tok_tformed = self._dropout(embedded_text[1][-1])
		tok_tformed = self.tok_feedforward(tok_tformed)
		tok_tformed = self.tok_layernorm(tok_tformed)
		logits = batch_cos_sim(tok_tformed, pooled_text).squeeze()
		n_batches = pooled_text.shape[0]
		assert n_batches % 2 == 0, 'There has to be an even number of inputs here'

		label_mask = (label < 0)
		updated_labels = list(range(n_batches))
		updated_labels = torch.tensor(updated_labels[(n_batches // 2):] + updated_labels[:(n_batches // 2)]).view(-1, 1)
		updated_labels = updated_labels.to(label.device)
		new_labels = torch.ones_like(label, device=label.device) * updated_labels
		new_labels = new_labels.masked_fill(label_mask, label.min())

		output_dict = {}
		if label is not None:
			logits = logits.view(-1, logits.shape[-1])
			label = new_labels.long().view(-1)
			loss = self._loss(logits, label)
			output_dict["loss_full"] = loss
			output_dict["loss"] = loss.sum() / (len(loss.nonzero()) + EPS)
			self._accuracy(logits, label)
		return output_dict
	
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return super().get_metrics(reset)

