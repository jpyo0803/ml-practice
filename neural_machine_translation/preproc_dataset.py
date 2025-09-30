import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'
token_transform = {}

'''
    Install the required spaCy models if not already installed:
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
'''

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def yield_tokens(data_iter, language_index):
    # 토큰 생성을 위한 헬퍼 함수
    for data_sample in data_iter:
        yield token_transform[SRC_LANGUAGE if language_index == 0 else TGT_LANGUAGE](data_sample[language_index])

def load_and_preprocess_nmt(batch_size=32, max_vocab_size=5000):
    print("Load and preprocess Multi30k dataset...")
    train_datapipe, val_datapipe, test_datapipe = Multi30k(root='./data', split=('train', 'valid', 'test'), language_pair=(TGT_LANGUAGE, SRC_LANGUAGE))

    print("Building vocabularies...")
    vocab_transform = {}

    # 언어 인덱스: 영어는 1, 독일어는 0
    en_vocab = build_vocab_from_iterator(yield_tokens(train_datapipe, 1),
                                            min_freq=2,
                                            specials=special_symbols,
                                            special_first=True,
                                            max_tokens=max_vocab_size)
    en_vocab.set_default_index(UNK_IDX)  # 알 수 없는 토큰은 <unk>로 매핑
    vocab_transform[SRC_LANGUAGE] = en_vocab

    de_vocab = build_vocab_from_iterator(yield_tokens(train_datapipe, 0),
                                            min_freq=2,
                                            specials=special_symbols,
                                            special_first=True,
                                            max_tokens=max_vocab_size)
    de_vocab.set_default_index(UNK_IDX)  # 알 수 없는 토큰은 <unk>로 매핑
    vocab_transform[TGT_LANGUAGE] = de_vocab

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    print(f"Unique tokens in source (en) vocabulary: {SRC_VOCAB_SIZE}")
    print(f"Unique tokens in target (de) vocabulary: {TGT_VOCAB_SIZE}")

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_tensor = torch.tensor([BOS_IDX] + vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](src_sample)) + [EOS_IDX], dtype=torch.long)
            tgt_tensor = torch.tensor([BOS_IDX] + vocab_transform[TGT_LANGUAGE](token_transform[TGT_LANGUAGE](tgt_sample)) + [EOS_IDX], dtype=torch.long)
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_dataloader = DataLoader(list(train_datapipe), batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(list(val_datapipe), batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(list(test_datapipe), batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, vocab_transform

if __name__ == "__main__":
    load_and_preprocess_nmt()