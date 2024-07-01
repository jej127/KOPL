import numpy as np
import json

if __name__ == '__main__':
    np.random.seed(123)
    with open('./data/kold_v1.json') as f:
        data = json.load(f)

    eval_idxs = np.random.choice(len(data), int(len(data)*0.2), replace=False)
    dev_idxs, test_idxs = eval_idxs[:int(len(eval_idxs)*0.5)], eval_idxs[int(len(eval_idxs)*0.5):]

    examples_train, examples_dev, examples_test = [],[],[]
    for idx, example in enumerate(data):
        ex_reduced = {'guid': example['guid'], 'comment': example['comment'], 'OFF': int(example['OFF'])}
        if idx in dev_idxs:
            examples_dev.append(ex_reduced)
        elif idx in test_idxs:
            examples_test.append(ex_reduced)
        else:
            examples_train.append(ex_reduced)

    with open("./data/kold_v1_split_train.json", "w") as f:
        json.dump(examples_train, f, indent="\t", ensure_ascii=False)

    with open("./data/kold_v1_split_val.json", "w") as f:
        json.dump(examples_dev, f, indent="\t", ensure_ascii=False)

    with open("./data/kold_v1_split_test.json", "w") as f:
        json.dump(examples_test, f, indent="\t", ensure_ascii=False)
