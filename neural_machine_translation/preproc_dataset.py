import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import spacy
from collections import Counter

SRC_LANGUAGE = "en"
TGT_LANGUAGE = "de"

# spaCy tokenizer 로드
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

token_transform = {
    SRC_LANGUAGE: lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)],
    TGT_LANGUAGE: lambda text: [tok.text.lower() for tok in spacy_de.tokenizer(text)],
}

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# ===== Vocab 클래스 =====
class Vocab:
    def __init__(self, counter, specials):
        # index to token string
        self.itos = list(specials) + [tok for tok, freq in counter.items() if tok not in specials]
        # token string to index
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

# ===== Vocab 빌더 =====
def build_vocab(dataset_split, language, min_freq=1, max_size=5000):
    counter = Counter()
    for example in dataset_split:
        # 특정 언어의 문장을 spaCy 토크나이저로 토큰화함
        tokens = token_transform[language](example[language])
        # 토큰 빈도수 업데이트
        counter.update(tokens)
    # 가장 빈도수가 높은 토큰들로 vocab 구성
    most_common = dict(counter.most_common(max_size))
    return Vocab(most_common, special_symbols)

# ===== 데이터 로더 =====
def load_and_preprocess_nmt(batch_size=32, max_vocab_size=5000):
    print("Load Multi30k dataset from HuggingFace (bentrevett/multi30k)...")
    dataset = load_dataset("bentrevett/multi30k")

    train_data, val_data, test_data = dataset["train"], dataset["validation"], dataset["test"]
    '''
        train_data, val_data, test_data는 각각 Dataset 객체
        (영어 문장, 독일어 문장) 쌍을 포함
    '''

    print("Building vocabularies...")
    vocab_transform = {}
    vocab_transform[SRC_LANGUAGE] = build_vocab(train_data, SRC_LANGUAGE, max_size=max_vocab_size)
    vocab_transform[TGT_LANGUAGE] = build_vocab(train_data, TGT_LANGUAGE, max_size=max_vocab_size)

    # Vocab transform은 언어들의 vocab을 저장하는 딕셔너리

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    print(f"Unique tokens in source (en) vocabulary: {SRC_VOCAB_SIZE}")
    print(f"Unique tokens in target (de) vocabulary: {TGT_VOCAB_SIZE}")

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for example in batch:
            # 문장을 토큰 리스트로 변환 후 vocab을 통해 인덱스 리스트로 변환
            src_tokens = vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](example[SRC_LANGUAGE]))
            tgt_tokens = vocab_transform[TGT_LANGUAGE](token_transform[TGT_LANGUAGE](example[TGT_LANGUAGE]))
            # 앞 뒤로 BOS, EOS 토큰 추가
            src_tensor = torch.tensor([BOS_IDX] + src_tokens + [EOS_IDX], dtype=torch.long)
            tgt_tensor = torch.tensor([BOS_IDX] + tgt_tokens + [EOS_IDX], dtype=torch.long)
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        # 패딩을 위해 시퀀스 길이를 맞춤
        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, vocab_transform

if __name__ == "__main__":
    train_dl, val_dl, test_dl, vocab_transform = load_and_preprocess_nmt()