import os
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import CLIPModel, CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CLIP_finetune_test(image_folder, ground_truth_file, model_ckpt, dst_folder):
    # image_folder = "cn"
    # ground_truth_file = "ground_truth_cn.json"
    # model_ckpt = "clip_finetuned.pth"

    print("loading")

    with open("ground_truth_cn.json", "r", encoding="utf-8") as f:
        full_gt = json.load(f)
    label_names = sorted(set(full_gt.values()))
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    NUM_CLASSES = len(label2id)

    # 2. Now read the actual ground truth file for this test
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    class CLIPFineTuner(nn.Module):
        def __init__(self, clip_model, num_classes):
            super().__init__()
            self.clip = clip_model
            self.classifier = nn.Linear(clip_model.config.projection_dim, num_classes)

        def forward(self, pixel_values):
            features = self.clip.get_image_features(pixel_values=pixel_values)
            return self.classifier(features)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.vision_model.post_layernorm.requires_grad = True
    clip_model.visual_projection.requires_grad = True

    model = CLIPFineTuner(clip_model, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    Ks = [1, 2, 3, 4, 5]
    topk_correct = {k: 0 for k in Ks}
    topk_y_pred = {k: [] for k in Ks}
    topk_y_true = {k: [] for k in Ks}

    all_topk_preds = []

    print("predicting")

    for fname, label_name in ground_truth.items():
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            print(f"Warning: image not found → {fname}")
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(inputs["pixel_values"])
            probs = logits.softmax(dim=1)[0]
            top5 = torch.topk(probs, 5)
            top5_ids = top5.indices.cpu().numpy()
            top5_scores = top5.values.cpu().numpy()
            top5_labels = [id2label[idx] for idx in top5_ids]

        all_topk_preds.append({
            "image": fname,
            "true_label": label_name,
            "top5_labels": top5_labels,
            "top5_scores": top5_scores
        })

        # 针对不同的 K 统计准确率和分类指标
        for k in Ks:
            topk = top5_labels[:k]
            topk_y_true[k].append(label_name)
            topk_y_pred[k].append(topk[0])   # top-1 always用第一个
            if label_name in topk:
                topk_correct[k] += 1
    
    # Overall macro metrics
    print("Overall macro metrics")
    y_true = topk_y_true[1]
    y_pred = topk_y_pred[1]

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
    df_overall_metrics.to_csv(dst_folder + "overall_macro_metrics.csv", index=False)

    # Per-class macro metrics
    print("Per-class macro metrics")
    y_true = topk_y_true[1]
    y_pred = topk_y_pred[1]

    p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average=None, zero_division=0)

    df_per_class_metrics = pd.DataFrame({
        "Class": label_names,
        "Precision": p_c,
        "Recall": r_c,
        "F1": f1_c,
        "Support": s_c
    })

    df_per_class_metrics.to_csv(dst_folder + "per_class_macro_metrics.csv", index=False)

    # Overall top-k accuracy
    print("Overall top-k accuracy")
    overall_acc = {"Top-K":[], "Accuracy":[]}
    for k in Ks:
        overall_acc["Top-K"].append(f"Top-{k}")
        overall_acc["Accuracy"].append(topk_correct[k] / len(all_topk_preds))
    df_overall_acc = pd.DataFrame(overall_acc)
    df_overall_acc.to_csv(dst_folder + "overall_top5_accuracy.csv", index=False)

    # Per-class top-5 accuracy table
    per_class_acc_table = []
    for class_name in label_names:
        row = [class_name]
        for k in Ks:
            y_true_c = pd.Series(topk_y_true[k])
            y_pred_c = pd.Series(topk_y_pred[k])
            mask = y_true_c == class_name
            support = mask.sum()
            if support == 0:
                row.append(0.0)
            else:
                acc = (y_pred_c[mask].values == y_true_c[mask].values).mean()
                row.append(acc)
        per_class_acc_table.append(row)
    header = ["Class"] + [f"Top-{k} Acc" for k in Ks]
    df_per_class_acc = pd.DataFrame(per_class_acc_table, columns=header)
    df_per_class_acc.to_csv(dst_folder + "per_class_top5_accuracy.csv", index=False)

    # Per-image top-5 predictions (labels and probabilities, and when the true label is found)
    print("Per-image top-5 predictions")
    top5_result_rows = []
    for item in all_topk_preds:
        row = {
            "Image": item['image'],
            "True Label": item['true_label']
        }
        for i, (lbl, prob) in enumerate(zip(item['top5_labels'], item['top5_scores']), 1):
            row[f"Top{i}_Label"] = lbl
            row[f"Top{i}_Prob"] = prob
        top5_result_rows.append(row)
    df_top5_per_image = pd.DataFrame(top5_result_rows)

    accurate_top = []
    for _, row in df_top5_per_image.iterrows():
        true_label = row["True Label"]
        found = -1
        for k in range(1, 6):
            if row[f"Top{k}_Label"] == true_label:
                found = k
                break
        accurate_top.append(found)

    df_top5_per_image["Rank"] = accurate_top
    df_top5_per_image.to_csv(dst_folder + "per_image_top5.csv", index=False)

CLIP_finetune_test(
    image_folder="cn",
    ground_truth_file="ground_truth_cn.json",
    model_ckpt="clip_finetuned.pth",
    dst_folder="rs/finetuneRS/cn/"
)

CLIP_finetune_test(
    image_folder="de",
    ground_truth_file="ground_truth_de.json",
    model_ckpt="clip_finetuned.pth",
    dst_folder="rs/finetuneRS/de/"
)

CLIP_finetune_test(
    image_folder="in",
    ground_truth_file="ground_truth_in.json",
    model_ckpt="clip_finetuned.pth",
    dst_folder="rs/finetuneRS/in/"
)

