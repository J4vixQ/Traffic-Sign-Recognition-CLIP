import json
import re
import os

def compare_categories(prompt_json_path, gt_json_path):
    # 读取 prompt.json
    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    prompt_cats = set(prompt_data.keys())

    # 读取 ground_truth.json
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    gt_cats = set(gt_data.values())

    # 找出只在 prompt.json 的类别
    only_in_prompt = prompt_cats - gt_cats
    # 找出只在 ground_truth.json 的类别
    only_in_gt = gt_cats - prompt_cats

    if only_in_prompt:
        print("Only in prompt.json:", only_in_prompt)
    if only_in_gt:
        print("Only in ground_truth.json:", only_in_gt)
    if not only_in_prompt and not only_in_gt:
        print("Categories are the same in both files.")

    # 返回结果方便调用
    return list(only_in_prompt), list(only_in_gt)

compare_categories('prompt_cn.json', 'ground_truth_cn.json')
compare_categories('prompt_de.json', 'ground_truth_de.json')
compare_categories('prompt_in.json', 'ground_truth_in.json')



def normalize_filenames(folder):
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            # 匹配类似 abc_def_0.png, mn_22.png, xx_8.png
            m = re.match(r'^(.+?)_(\d+)\.png$', fname)
            if m:
                prefix, num = m.groups()
                if len(num) < 3:
                    new_fname = f"{prefix}_{int(num):03d}.png"
                    src = os.path.join(folder, fname)
                    dst = os.path.join(folder, new_fname)
                    # 避免覆盖同名
                    if not os.path.exists(dst):
                        os.rename(src, dst)
                        print(f"Renamed: {fname} → {new_fname}")
                    else:
                        print(f"Skip: {fname} (target {new_fname} already exists)")

normalize_filenames('cn_rl')
normalize_filenames('cn_ai')