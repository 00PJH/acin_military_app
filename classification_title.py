import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer():
    model_name = "00PJH/klue-bert-tunning-classification-military-news"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def batch_inference(df_input, tokenizer, model, batch_size=64):
    texts = df_input['title'].tolist()
    all_predictions = []
    all_probabilities = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"총 {len(texts)}건 기사 제목 분류 중... (총 {total_batches}개 배치)")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=128
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            # 국방 기사(label 1) 확률 저장
            defense_probs = probs[:, 1].cpu().numpy().tolist()
            all_probabilities.extend(defense_probs)

            if (i // batch_size + 1) % 50 == 0 or (i + batch_size) >= len(texts):
                print(f"  분류 진행률: {min(i + batch_size, len(texts))}/{len(texts)} ({(i // batch_size + 1)/total_batches*100:.1f}%)")

    df_result = df_input.copy()
    df_result['predictions'] = all_predictions
    df_result['defense_prob'] = [round(p * 100, 2) for p in all_probabilities]
    return df_result

def run(input_data, batch_size=64):
    tokenizer, model = load_model_and_tokenizer()
    df_classified = batch_inference(input_data, tokenizer, model, batch_size=batch_size)
    filtered_df = df_classified[df_classified['predictions'] == 1].copy()
    return filtered_df
