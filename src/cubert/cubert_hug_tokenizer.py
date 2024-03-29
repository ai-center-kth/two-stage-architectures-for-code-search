# Borrowed from https://gist.github.com/DNGros/7c2fa0dcf566bd9f3732618669b591dd

import os
import collections
from typing import *
from transformers import BertTokenizer
from .cubert_tokenizer import CuBertTokenizer
from .python_tokenizer import PythonTokenizer
from tensor2tensor.data_generators import text_encoder

def combine_tokenizer_with_subword(
    initial_tokenizer: CuBertTokenizer,
    subword_tokenizer: text_encoder.SubwordTextEncoder
) -> Callable[[str], List[str]]:
    # Try to match the functionality at
    # https://github.com/google-research/google-research/blob/50c6cd94b5/cubert/code_to_subtokenized_sentences.py#L111-L118
    def tokenize(string: str) -> List[str]:
        toks = initial_tokenizer.tokenize(string)
        return flatten_list(
            subword_tokenizer.decode_list(
                subword_tokenizer.encode_without_tokenizing(token)
            )
            for token in toks
        )
    return tokenize


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class CuBertHugTokenizer(BertTokenizer):
    # A hacky version that seems to work at least for python
    def __init__(
        self,
        vocab_file: str
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token="[UNK]_",
            sep_token="[SEP]_",
            pad_token="<pad>_",
            cls_token="[CLS]_",
            mask_token="[MASK]_"
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    vocab_file)
            )
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.first_tokenizer = PythonTokenizer(50_000)
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(str(vocab_file))
        self._combined_func = combine_tokenizer_with_subword(
            self.first_tokenizer, self.subword_tokenizer)

    @property
    def do_lower_case(self):
        return False

    def _tokenize(self, text):
        return self._combined_func(text)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def _convert_token_to_id(self, token):
        return self.subword_tokenizer._subtoken_string_to_id[token]
    
    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab