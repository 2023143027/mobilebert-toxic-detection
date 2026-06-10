import torch
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import numpy as np


def predict_comments(text_list):
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 방금 전 학습 완료되어 저장된 모델과 토크나이저 경로 지정
    model_path = "./mobilebert_toxic"

    tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased")
    model = MobileBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 예측용 6개 라벨 매핑 테이블
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    print("\n=== 테스트 댓글 멀티 라벨 추론 시작 ===")

    for text in text_list:
        # 토큰화 진행
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"  # 파이토치 텐서 형태로 즉시 반환
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 모델 예측
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        # 학습 때 사용한 시그모이드 및 임계값 0.5 적용
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

        # 결과 출력
        print(f"\n[입력 문장]: {text}")
        active_labels = []
        for i, idx in enumerate(preds):
            if idx == 1:
                active_labels.append(f"{label_cols[i]}(확률: {probs[i] * 100:.1f}%)")

        if active_labels:
            print(f"-> 탐지된 악성 성향: {', '.join(active_labels)}")
        else:
            print("->탐지된 악성 성향: 정상 (Clean)")


if __name__ == "__main__":
    # 검증해보고 싶은 예시 문장들 (실제 영어 악성 댓글 패턴 테스트)
    sample_texts = [
        "Hey, that is a really good point! Thank you for sharing this information.",
        "You are so stupid and idiot, go away from here right now!",
        "I will find you and I will hurt you badly, watch your back.",
        "This disgusting person belongs to a trash group, I hate their entire race."
    ]

    predict_comments(sample_texts)