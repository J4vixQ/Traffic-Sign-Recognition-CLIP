import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

with open("ground_truth_cn.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

with open("prompt_cn.json", "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

CATEGORIES = list(prompt_data.keys())
PROMPT_VARIATIONS = list(prompt_data.values())

def init_text_features(variations: list, device: str):
    """
    把所有 prompt 一次性编码为归一化后的特征向量。
    返回：
      text_features: Tensor[num_prompts, dim]
      prompt_to_cat: List[num_prompts]，映射到类别索引
    """
    all_prompts = [p for vs in variations for p in vs]
    prompt_to_cat = []
    for idx, vs in enumerate(variations):
        prompt_to_cat += [idx] * len(vs)

    inputs = processor(
        text=all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats, prompt_to_cat

# 执行一次
TEXT_FEATURES, PROMPT_TO_CAT = init_text_features(PROMPT_VARIATIONS, DEVICE)

def classify_image(img: Image.Image, 
                   text_feats: torch.Tensor, 
                   prompt_to_cat: list, 
                   top_k: int = 5, 
                   return_all: bool = False):
    inputs = processor(images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feat = model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * img_feat @ text_feats.T
        probs = logits.softmax(dim=-1)[0].cpu().numpy()

    # 聚合到类别
    cat_probs = np.zeros(len(CATEGORIES))
    for i, p in enumerate(probs):
        cat_probs[prompt_to_cat[i]] += p
    cat_probs /= cat_probs.sum()

    if return_all:
        idxs = np.argsort(-cat_probs)
    else:
        idxs = np.argpartition(cat_probs, -top_k)[-top_k:]
        idxs = idxs[np.argsort(-cat_probs[idxs])]

    return [(CATEGORIES[i], float(cat_probs[i])) for i in idxs]


def classify_batch(imgs: list, **kwargs) -> list:
    return [ classify_image(img, TEXT_FEATURES, PROMPT_TO_CAT, **kwargs) for img in imgs ]

def evaluate_directory(directory: str,
                       ground_truth: dict,
                       batch_size: int = 16,
                       top_k: int = 5) -> pd.DataFrame:
    """
    return every image's top-5 predictions (labels and probabilities) and rank.
    """
    files = [f for f in os.listdir(directory) if f.lower().endswith((".png"))]
    rows = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        imgs  = []
        for fn in batch:
            img = Image.open(os.path.join(directory, fn)).convert("RGB")
            imgs.append(img)

        preds_batch = classify_batch(imgs, top_k=top_k, return_all=True)
        for fn, preds in zip(batch, preds_batch):
            true = ground_truth.get(fn, "Unknown")
            # extract top-5 predictions
            labels = [x[0] for x in preds[:5]]
            scores = [x[1] for x in preds[:5]]
            if len(labels) < 5:
                labels += [""] * (5 - len(labels))
                scores += [0.0] * (5 - len(scores))
            row = {
                "Image": fn,
                "True Label": true
            }
            # include top-5 predictions (labels and probabilities)
            for i in range(5):
                row[f"Top{i+1}_Label"] = labels[i]
                row[f"Top{i+1}_Prob"] = scores[i]
            # calculate rank
            found = -1
            for k in range(1, 6):
                if row[f"Top{k}_Label"] == true:
                    found = k
                    break
            row["Rank"] = found
            rows.append(row)
    return pd.DataFrame(rows)



def per_image_top5(ground_truth, source_dir, save_path):
    with open(ground_truth, "r", encoding="utf-8") as f:
        gt = json.load(f)
    df_results = evaluate_directory(source_dir, gt, batch_size=16, top_k=5)
    df_results.to_csv(save_path, index=False)
    return df_results

def overall_macro_metrics(df_results, ground_truth, save_path):
    # Overall macro metrics
    with open(ground_truth, "r", encoding="utf-8") as f:
        gt = json.load(f)

    label_names = sorted(set(gt.values()))
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    NUM_CLASSES = len(label2id)

    y_true = df_results["True Label"].values
    y_pred = df_results["Top1_Label"].values

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average='macro', zero_division=0)

    overall_metrics = {
        "Accuracy": [acc],
        "Macro Precision": [p],
        "Macro Recall": [r],
        "Macro F1": [f1],
        "Support": [len(y_true)]
    }
    df_overall_metrics = pd.DataFrame(overall_metrics)
    df_overall_metrics.to_csv(save_path, index=False)

def overall_top5_accuracy(df_results, save_path):
    Ks = [1, 2, 3, 4, 5]
    overall_acc = {}

    for k in Ks:
        hit = 0
        for i, row in df_results.iterrows():
            found = False
            for ki in range(1, k+1):
                if row[f"Top{ki}_Label"] == row["True Label"]:
                    found = True
                    break
            if found:
                hit += 1
        overall_acc[f"Top-{k}"] = hit / len(df_results)

    overall_acc_df = pd.DataFrame(list(overall_acc.items()), columns=["Top-K", "Accuracy"])
    overall_acc_df.to_csv(save_path, index=False)
    return

def per_class_macro_metrics(df_results, ground_truth, save_path):
    with open(ground_truth, "r", encoding="utf-8") as f:
        gt = json.load(f)

    label_names = sorted(set(gt.values()))
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    NUM_CLASSES = len(label2id)

    y_true = df_results["True Label"].values
    y_pred = df_results["Top1_Label"].values

    p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average=None, zero_division=0)

    df_per_class_metrics = pd.DataFrame({
        "Class": label_names,
        "Precision": p_c,
        "Recall": r_c,
        "F1": f1_c,
        "Support": s_c
    })
    df_per_class_metrics.to_csv(save_path, index=False)

def per_class_top5_accuracy(df_results, ground_truth, save_path):
    with open(ground_truth, "r", encoding="utf-8") as f:
        gt = json.load(f)

    label_names = sorted(set(gt.values()))
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    NUM_CLASSES = len(label2id)

    Ks = [1, 2, 3, 4, 5]
    per_class_acc = []

    for class_name in label_names:
        row = [class_name]
        mask = df_results["True Label"] == class_name
        df_sub = df_results[mask]
        n = len(df_sub)
        for k in Ks:
            hit = 0
            for _, r in df_sub.iterrows():
                found = False
                for ki in range(1, k+1):
                    if r[f"Top{ki}_Label"] == r["True Label"]:
                        found = True
                        break
                if found:
                    hit += 1
            acc = hit / n if n > 0 else 0.0
            row.append(acc)
        per_class_acc.append(row)

    header = ["Class"] + [f"Top-{k} Acc" for k in Ks]
    df_per_class_acc = pd.DataFrame(per_class_acc, columns=header)
    df_per_class_acc.to_csv(save_path, index=False)



df_de = per_image_top5("ground_truth_de.json", "de", "rs/zeroshotRS/de/per_image_top5.csv")
overall_macro_metrics(df_de, "ground_truth_de.json", "rs/zeroshotRS/de/overall_macro_metrics.csv")
overall_top5_accuracy(df_de, "rs/zeroshotRS/de/overall_top5_accuracy.csv")
per_class_macro_metrics(df_de, "ground_truth_de.json", "rs/zeroshotRS/de/per_class_macro_metrics.csv")
per_class_top5_accuracy(df_de, "ground_truth_de.json", "rs/zeroshotRS/de/per_class_top5_accuracy.csv")

df_in = per_image_top5("ground_truth_in.json", "in", "rs/zeroshotRS/in/per_image_top5.csv")
overall_macro_metrics(df_in, "ground_truth_in.json", "rs/zeroshotRS/in/overall_macro_metrics.csv")
overall_top5_accuracy(df_in, "rs/zeroshotRS/in/overall_top5_accuracy.csv")
per_class_macro_metrics(df_in, "ground_truth_in.json", "rs/zeroshotRS/in/per_class_macro_metrics.csv")
per_class_top5_accuracy(df_in, "ground_truth_in.json", "rs/zeroshotRS/in/per_class_top5_accuracy.csv")