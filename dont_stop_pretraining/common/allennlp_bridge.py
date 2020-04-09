import codecs
import logging
import os
from typing import Iterable

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.common.util import namespace_match
from allennlp.data import instance as adi  # pylint: disable=unused-import
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


@Vocabulary.register("extended_vocabulary")
class ExtendedVocabulary(Vocabulary):
    """
    Hack to augment the allennlp Vocabulary with serialization directory so we can dump supermasks
    """

    @overrides
    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.
        Parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        self.serialization_dir = "/".join(directory.split('/')[:-1])  # pylint: disable=W0201
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning("vocabulary serialization directory %s is not empty", directory)

        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        for namespace, mapping in self._index_to_token.items():
            # Each namespace gets written to its own file, in index order.
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    print(mapping[i].replace('\n', '@@NEWLINE@@'), file=token_file)
