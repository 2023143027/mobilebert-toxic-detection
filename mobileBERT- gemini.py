import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
#  시각화 차트 생성을 위한 라이브러리 추가
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치:", device)

    logging.set_verbosity_error()

    path = "train.csv"
    df = pd.read_csv(path)

    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df = df[["comment_text"] + label_cols].dropna()

    #  데이터 볼륨을 35,000건으로 대폭 확장 (최소 25,000건 이상 조건)
    df = df.head(35000)

    plt.figure(figsize=(7, 4))
    df[label_cols].sum().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Label Distribution (35,000 samples)')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('data_dist_plot.png')
    plt.close()

    text = list(df["comment_text"].values)
    labels = df[label_cols].values

    print("\n=== 데이터 확인 ===")
    print("문장:", text[:2])
    print("라벨:\n", labels[:2])

    print("\n=== 데이터 개수 확인 ===")
    print("사용한 데이터 개수:", len(df))

    print("\n=== 라벨별 악성 댓글 개수 확인 ===")
    print(df[label_cols].sum())

    print("\n=== 라벨별 비율 확인 (%) ===")
    print((df[label_cols].mean() * 100).round(2))

    tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased")

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
        padding="max_length"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("\n=== 토큰화 샘플 ===")
    for j in range(3):
        print(f"\n{j + 1}번째 데이터")
        print("토큰:", input_ids[j][:20])
        print("어텐션 마스크:", attention_mask[j][:20])

    tx, vx, ty, vy = train_test_split(
        input_ids,
        labels,
        test_size=0.2,
        random_state=2026
    )

    tm, vm, _, _ = train_test_split(
        attention_mask,
        labels,
        test_size=0.2,
        random_state=2026
    )

    batch_size = 8

    train_inputs = torch.tensor(tx)
    train_labels = torch.tensor(ty, dtype=torch.float)
    train_masks = torch.tensor(tm)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

    valid_inputs = torch.tensor(vx)
    valid_labels = torch.tensor(vy, dtype=torch.float)
    valid_masks = torch.tensor(vm)

    valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler)

    model = MobileBertForSequenceClassification.from_pretrained(
        "mobilebert-uncased",
        num_labels=6,
        problem_type="multi_label_classification"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8
    )

    epoch = 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * epoch
    )

    epoch_results = []

    for e in range(epoch):
        model.train()
        total_train_loss = 0.0

        process_bar = tqdm(
            train_dataloader,
            desc=f"Train Epoch {e + 1}",
            leave=False
        )

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch

            model.zero_grad()

            outputs = model(
                batch_ids,
                attention_mask=batch_masks,
                labels=batch_labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            process_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()

        train_preds = []
        train_true = []

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch

            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)

            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).int()

            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())

        train_acc = np.mean(np.array(train_preds) == np.array(train_true))

        valid_preds = []
        valid_true = []

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch

            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)

            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).int()

            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())

        valid_acc = np.mean(np.array(valid_preds) == np.array(valid_true))

        epoch_results.append([avg_train_loss, train_acc, valid_acc])

        print("\n=== 학습 및 검증 결과 ===")
        for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
            print(
                f"Epoch {idx}: "
                f"학습오차={loss:.4f}, "
                f"학습정확도={tacc:.4f}, "
                f"검증정확도={vacc:.4f}"

            )

    print("\n=== 모델 저장 ===")
    model.save_pretrained("mobilebert_toxic")
    print("모델 저장 완료")

    print("\n=== 최종 프로젝트 성능 판정 ===")
    final_vacc = epoch_results[-1][2]
    print(f"최종 에포크 검증 정확도: {final_vacc:.4f}")
    if final_vacc >= 0.85:
        print("판정: 검증 데이터 정확도가 0.85 이 상이므로 성공적으로 미세조정(Fine-Tuning)되었습니다.")
    else:
        print("판정: 정확도가 0.85 미만이므로 데이터 정제 또는 추가 학습이 필요합니다.")


    # [수업 요구사항] 보고서 첨부용 Loss 및 Accuracy 시각화 차트 자동 저장 파트
    epochs_range = range(1, epoch + 1)
    losses = [x[0] for x in epoch_results]
    train_accs = [x[1] for x in epoch_results]
    valid_accs = [x[2] for x in epoch_results]

    # 1. Loss 차트 생성 및 저장
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, losses, marker='o', color='red', label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # 2. Accuracy 차트 생성 및 저장
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_accs, marker='o', color='blue', label='Train Acc')
    plt.plot(epochs_range, valid_accs, marker='s', color='green', label='Valid Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    print("\n=== 시각화 결과 저장 완료 ===")
    print("프로젝트 폴더 내 'loss_plot.png' 및 'accuracy_plot.png' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()