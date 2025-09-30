import torch
from preproc_dataset import token_transform, load_and_preprocess_nmt, BOS_IDX, EOS_IDX, SRC_LANGUAGE, TGT_LANGUAGE
from encoder import Encoder
from attention import Attention
from decoder import Decoder
from seq2seq import Seq2Seq

def translate_sentence(sentence, model, vocabs, device, max_len=50):
    model.eval()

    # 문장 토큰화 및 인덱싱
    tokens = [token.lower() for token in token_transform[SRC_LANGUAGE](sentence)]
    tokens = [BOS_IDX] + [vocabs[SRC_LANGUAGE][token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [BOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break
    
    # 인덱스를 단어로 변환
    trg_tokens = [vocabs[TGT_LANGUAGE].get_itos()[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1]) # <bos>와 <eos> 제외


if __name__ == '__main__':
    # 모델과 단어장 로드를 위해 데이터로더를 한번 실행해야 합니다.
    # 실제 어플리케이션에서는 단어장을 별도로 저장하고 불러오는 것이 좋습니다.
    _, _, _, vocabs = load_and_preprocess_nmt(max_vocab_size=5000)

    # 하이퍼파라미터 (훈련 스크립트와 동일해야 함)
    INPUT_DIM = len(vocabs[SRC_LANGUAGE])
    OUTPUT_DIM = len(vocabs[TGT_LANGUAGE])
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 구조 초기화
    attn = Attention(HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    # 훈련된 가중치 로드
    model_path = "./model_weight.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 번역할 문장
    example_sentence = "I like apple."
    
    print(f"Source: {example_sentence}")
    translation = translate_sentence(example_sentence, model, vocabs, device)
    print(f"Translated: {translation}")