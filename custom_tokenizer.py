from transformers import PreTrainedTokenizer
from pathlib import Path
from typing import Optional, Tuple, Dict
import json
import torch
from torch.utils.data import Dataset, DataLoader

class FixedVocabTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = None):
        self.__token_ids = vocab
        super().__init__(max_len=max_len)
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}

    def _tokenize(self, text: str, **kwargs):
        return text.split(' ')

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        json.dump(self.__token_ids, open(vocab_path, 'w'))
        return str(vocab_path),

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)
    

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        # or use the RobertaTokenizer from `transformers` directly.
        self.examples = []
        # For every value in the dataframe 
        for example in df.values:
            # 
            x=tokenizer.encode_plus(example, max_length = max_len, truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])