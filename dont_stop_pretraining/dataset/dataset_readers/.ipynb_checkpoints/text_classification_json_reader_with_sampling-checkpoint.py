import json
import logging
from io import TextIOWrapper
from typing import Dict
import numpy as np
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField, Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text_classification_json_with_sampling")
class TextClassificationJsonReaderWithSampling(TextClassificationJsonReader):
    """
    Reads tokens and (optionally) their labels from a from text classification dataset.

    This dataset reader inherits from TextClassificationJSONReader, but differs from its parent
    in that it is primed for semisupervised learning. This dataset reader allows for:
        3) Throttling the training data to a random subsample (according to the numpy seed),
           for analysis of the effect of semisupervised models on different amounts of labeled
           data

    Expects a "tokens" field and a "label" field in JSON format.

    The output of ``read`` is a list of ``Instances`` with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``, if not ignoring labels.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to split the input text into words or other kinds of tokens.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    ignore_labels: ``bool``, optional (default = ``False``)
        If specified, will ignore labels when reading data.
    sample: ``int``, optional (default = ``None``)
        If specified, will sample data to a specified length.
            **Note**:
                1) This operation will *not* apply to any additional unlabeled data
                   (specified in `additional_unlabeled_data_path`).
                2) To produce a consistent subsample of data, use a consistent seed in your
                   training config.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 max_sequence_length: int = None,
                 sample: int = None,
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy,
                         token_indexers=token_indexers,
                         tokenizer=tokenizer,
                         max_sequence_length=max_sequence_length,
                         skip_label_indexing=skip_label_indexing)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sample = sample
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @staticmethod
    def _reservoir_sampling(file_: TextIOWrapper, sample: int):
        """
        A function for reading random lines from file without loading the
        entire file into memory.

        For more information, see here: https://en.wikipedia.org/wiki/Reservoir_sampling

        To create a k-length sample of a file, without knowing the length of the file in advance,
        we first create a reservoir array containing the first k elements of the file. Then, we further
        iterate through the file, replacing elements in the reservoir with decreasing probability.

        By induction, one can prove that if there are n items in the file, each item is sampled with probability
        k / n.

        Parameters
        ----------
        file : `_io.TextIOWrapper` - file path
        sample_size : `int` - size of random sample you want

        Returns
        -------
        result : `List[str]` - sample lines of file
        """
        # instantiate file iterator
        file_iterator = iter(file_)

        try:
            # fill the reservoir array
            result = [next(file_iterator) for _ in range(sample)]
        except StopIteration:
            raise ConfigurationError(f"sample size {sample} larger than number of lines in file.")

        # replace elements in reservoir array with decreasing probability
        for index, item in enumerate(file_iterator, start=sample):
            sample_index = np.random.randint(0, index)
            if sample_index < sample:
                result[sample_index] = item

        for line in result:
            yield line

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            if self._sample is not None:
                data_file = self._reservoir_sampling(data_file, self._sample)
            for line in data_file:
                items = json.loads(line)
                text = items["text"]
                label = str(items.get('label'))
                if text:
                    instance = self.text_to_instance(text=text, label=label)
                    yield instance

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
            tokens = self._truncate(tokens)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label,
                                         skip_indexing=self._skip_label_indexing)
        return Instance(fields)
