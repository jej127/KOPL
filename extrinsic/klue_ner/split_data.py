import numpy as np
from utils import create_examples_from_original_data

if __name__ == '__main__':
    np.random.seed(123)

    file_path = "data/klue-ner-v1.1_train.tsv"
    examples, _ = create_examples_from_original_data(file_path)

    dev_idxs = np.random.choice(len(examples), int(len(examples)*0.1), replace=False)

    examples_train, examples_dev = [],[]
    for idx, ex in enumerate(examples):
        if idx in dev_idxs:
            examples_dev.append(ex)
        else:
            examples_train.append(ex)

    with open("./data/train_split.txt", "w") as file:
        for ex in examples_train:
            for word, label in zip(ex.text_a, ex.label):
                file.write(word + " " + label + "\n")
            file.write("\n")

    with open("./data/dev_split.txt", "w") as file:
        for ex in examples_dev:
            for word, label in zip(ex.text_a, ex.label):
                file.write(word + " " + label + "\n")
            file.write("\n")
