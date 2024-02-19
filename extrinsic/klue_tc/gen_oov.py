import numpy as np
from attacks import select_attack
import random
from utils import create_examples
import json

def main():
    random.seed(123)
    np.random.seed(123)

    test_path = './data/ynat-v1.1_dev.json'
    examples = create_examples(test_path)

    for r in [0.1,0.3,0.5,0.7,0.9]:
        data = []
        for ex in examples:
            sentence = ex.text_a.strip().split()
            sentence_corrupted = ' '.join([select_attack(word,r) for word in sentence])
            data.append({"guid": ex.guid, "title": sentence_corrupted, "label": ex.label})


        with open(f"./data/ynat-v1.1_dev_{int(r*100)}.json", "w") as f:
            json.dump(data, f, indent="\t", ensure_ascii=False)


if __name__ == '__main__':
    main()