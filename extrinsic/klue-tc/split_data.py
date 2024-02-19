import numpy as np
import json
from utils import create_examples

if __name__ == '__main__':
    np.random.seed(123)

    data_path = './data/ynat-v1.1_train.json'
    examples = create_examples(data_path)

    dev_idxs = np.random.choice(len(examples), int(len(examples)*0.1), replace=False)

    train_data, val_data = [],[]
    for idx, ex in enumerate(examples):
        if idx in dev_idxs:
            val_data.append({"guid": ex.guid, "title": ex.text_a, "label": ex.label})
        else:
            train_data.append({"guid": ex.guid, "title": ex.text_a, "label": ex.label})

    with open(f"./data/ynat-v1.1_split_train.json", "w") as f:
        json.dump(train_data, f, indent="\t", ensure_ascii=False)
    with open(f"./data/ynat-v1.1_split_val.json", "w") as f:
        json.dump(val_data, f, indent="\t", ensure_ascii=False)

