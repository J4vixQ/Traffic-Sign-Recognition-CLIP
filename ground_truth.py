import os
import json
import csv

def generate_ground_truth(data_dir, save_as="ground_truth.json", save_format="json"):
    """
    Generate ground truth mapping from image filenames to class labels.

    Args:
        data_dir (str): Directory containing traffic sign images.
        save_as (str): Output file name.
        save_format (str): "json" or "csv".
    """
    ground_truth = {}

    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            # remove extension
            name_no_ext = os.path.splitext(fname)[0]
            # remove trailing _001 or _000
            label_parts = name_no_ext.split("_")[:-1]
            label = " ".join(label_parts).lower()
            ground_truth[fname] = label

    # Save as JSON or CSV
    if save_format == "json":
        with open(save_as, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=4, ensure_ascii=False)
        print(f"Ground truth saved to {save_as}")
    
    elif save_format == "csv":
        with open(save_as, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for k, v in ground_truth.items():
                writer.writerow([k, v])
        print(f"Ground truth saved to {save_as}")
    
    else:
        raise ValueError("save_format must be 'json' or 'csv'")

    return ground_truth

ground_truth = generate_ground_truth(
    data_dir="dataset_cn",
    save_as="ground_truth_cn.json",
    save_format="json"
)
ground_truth = generate_ground_truth(
    data_dir="dataset_de",
    save_as="ground_truth_de.json",
    save_format="json"
)
ground_truth = generate_ground_truth(
    data_dir="dataset_in",
    save_as="ground_truth_in.json",
    save_format="json"
)