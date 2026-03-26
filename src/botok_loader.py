# ref: https://github.com/explosion/spaCy/discussions/12932
# ref2: https://support.prodi.gy/t/data-to-spacy-is-not-using-my-custom-tokenizer/6522

import spacy
from spacy.tokens import Doc
from botok import WordTokenizer
from botok.config import Config
import pickle

import os as _os

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DIALECT_PACKS = _os.path.join(_REPO_ROOT, "pybo", "dialect_packs")

nlp = spacy.blank("bo")

class BoTokTokenizer:
    def __init__(self, nlp):
        config = Config(dialect_name="custom", base_path=str(_DIALECT_PACKS))
        self.wt = WordTokenizer(config=config)
        self.vocab = nlp.vocab

    def __call__(self, text):
        # tokens = self.wt.tokenize(text, split_affixes=False) # This line could be activated, instead of the following line.
        tokens = self.wt.tokenize(text, split_affixes=True)
        words = [token.text for token in tokens]
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

    # added
    def to_bytes(self):
        return pickle.dumps(self.__dict__)

    # added
    def from_bytes(self, data):
        self.__dict__.update(pickle.loads(data))

    # added
    def to_disk(self, path, **kwargs):
        with open(path, 'wb') as file_:
            file_.write(self.to_bytes())

    # added
    def from_disk(self, path, **kwargs):
        with open(path, 'rb') as file_:
            self.from_bytes(file_.read())

@spacy.registry.tokenizers("botok_tokenizer")
def create_botok_tokenizer():
    def create_tokenizer(nlp):
        return BoTokTokenizer(nlp)

    return create_tokenizer