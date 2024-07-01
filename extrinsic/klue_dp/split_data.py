import numpy as np
from utils import create_examples
from collections import Counter
import csv

if __name__ == '__main__':
    np.random.seed(123)

    data_path = './data/klue-dp-v1.1_train.tsv'
    examples = create_examples(data_path)
    c = Counter([ex.sent_id for ex in examples])
    data = []
    start_idx = 0
    for i in range(len(c)):
        token_ids, tokens, poses, heads, deps = [],[],[],[],[]
        word_length = c[i]
        guid, sent_id = examples[start_idx].guid, examples[start_idx].sent_id
        for j in range(start_idx, start_idx+word_length):
            token_ids.append(str(examples[j].token_id))
            tokens.append(examples[j].token.strip())
            poses.append(examples[j].pos)
            heads.append(examples[j].head)
            deps.append(examples[j].dep)

        data.append({"guid": guid, "text": ' '.join(tokens), "sent_id": sent_id, "token_id": token_ids, "token": tokens,
                    "pos": poses, "head": heads, "dep": deps})
        start_idx += word_length

    dev_idxs = np.random.choice(len(c), int(len(c)*0.1), replace=False)

    train_data, val_data = [],[]
    for d in data:
        if d["sent_id"] in dev_idxs:
            val_data.append(d)
        else:
            train_data.append(d)

    with open('./data/klue-dp-v1.1_split_train.tsv', 'w', encoding='utf-8', newline='') as f:
        for d in train_data:
            tw = csv.writer(f, delimiter='\t')
            tw.writerow(["## " + d["guid"], d["text"]])
            for token_id, token, pos, head, dep in zip(d["token_id"], d["token"], d["pos"], d["head"], d["dep"]):
                tw.writerow([token_id, token, token, pos, head, dep])
            tw.writerow("")
    with open('./data/klue-dp-v1.1_split_val.tsv', 'w', encoding='utf-8', newline='') as f:
        for d in val_data:
            tw = csv.writer(f, delimiter='\t')
            tw.writerow(["## " + d["guid"], d["text"]])
            for token_id, token, pos, head, dep in zip(d["token_id"], d["token"], d["pos"], d["head"], d["dep"]):
                tw.writerow([token_id, token, token, pos, head, dep])
            tw.writerow("")
