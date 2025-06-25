import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import json

def visual_per_class(data, savepath):
    colors = [(1, 0, 0), (1, 1, 0.5), (0, 1, 0)]  # red, orange, green
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # map accuracy 0~1 to red, orange, green gradient
    acc_cols = [col for col in data.columns if col.startswith("Top-") and col.endswith("Acc")]
    vis = data.style.background_gradient(subset=acc_cols, cmap=cmap, vmin=0, vmax=1)
    vis = vis.set_properties(**{'color': 'black'}, subset=acc_cols)

    # save
    vis.to_excel(savepath, index=False)
    return vis

def rgb_to_hex(color):
    rgb_int = (color * 255).astype(int)
    return '#{:02X}{:02X}{:02X}'.format(*rgb_int)

def make_green_colormap(prob):
    white = np.array([1, 1, 1])
    green = np.array([0, 1, 0])
    alpha = np.clip(prob, 0, 1)
    color = (1 - alpha) * white + alpha * green
    return f"background-color: {rgb_to_hex(color)}; color: black"

def make_red_colormap(prob):
    white = np.array([1, 1, 1])
    red = np.array([1, 0, 0])
    alpha = np.clip(prob, 0, 1)
    color = (1 - alpha) * white + alpha * red
    return f"background-color: {rgb_to_hex(color)}; color: black"

def visual_per_image(data, savepath):
    label_cols = [f"Top{i}_Label" for i in range(1, 6)]
    prob_cols = [f"Top{i}_Prob" for i in range(1, 6)]

    label_df = data[["Image", "True Label"] + label_cols].copy()
    prob_df = data[["Image", "True Label"] + prob_cols].copy()

    df_vis = label_df.copy()
    for col in prob_cols:
        df_vis[col] = prob_df[col]

    # compare labels with truth to determine colors
    def style_func(row):
        styles = []
        true_label = row["True Label"]
        for i in range(1, 6):
            pred_label = row[f"Top{i}_Label"]
            prob = row[f"Top{i}_Prob"]
            if pred_label == true_label:
                styles.append(make_green_colormap(prob))
            else:
                styles.append(make_red_colormap(prob))
        return pd.Series(styles, index=label_cols)

    styled = df_vis.style.apply(style_func, axis=1)
    final_cols = ["Image", "True Label"] + label_cols
    styled = styled.hide(axis="columns", subset=[col for col in df_vis.columns if col not in final_cols])

    # save
    styled.to_excel(savepath, index=False)
    return styled

def visualizeRS(perclass_file_path, per_class_save_path, per_image_file_path, per_image_save_path):
    perclass = pd.read_csv(perclass_file_path)
    visual_pc = visual_per_class(perclass, per_class_save_path)
    perimage = pd.read_csv(per_image_file_path)
    visual_pi = visual_per_image(perimage, per_image_save_path)
    return

perclass_file_path = 'rs/finetuneRS/cn/per_class_top5_accuracy.csv'
per_class_save_path = 'rs/finetuneRS/cn/per_class_top5_accuracy.xlsx'
per_image_file_path = 'rs/finetuneRS/cn/per_image_top5.csv'
per_image_save_path = 'rs/finetuneRS/cn/per_image_top5.xlsx'

visualizeRS(perclass_file_path, per_class_save_path, per_image_file_path, per_image_save_path)

perclass_file_path = 'rs/finetuneRS/de/per_class_top5_accuracy.csv'
per_class_save_path = 'rs/finetuneRS/de/per_class_top5_accuracy.xlsx'
per_image_file_path = 'rs/finetuneRS/de/per_image_top5.csv'
per_image_save_path = 'rs/finetuneRS/de/per_image_top5.xlsx'

visualizeRS(perclass_file_path, per_class_save_path, per_image_file_path, per_image_save_path)

perclass_file_path = 'rs/finetuneRS/in/per_class_top5_accuracy.csv'
per_class_save_path = 'rs/finetuneRS/in/per_class_top5_accuracy.xlsx'
per_image_file_path = 'rs/finetuneRS/in/per_image_top5.csv'
per_image_save_path = 'rs/finetuneRS/in/per_image_top5.xlsx'

visualizeRS(perclass_file_path, per_class_save_path, per_image_file_path, per_image_save_path)

