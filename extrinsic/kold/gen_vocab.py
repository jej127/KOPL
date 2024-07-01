from utils import WordVocabulary

train_path = './data/kold_v1_split_train.json'
dev_path = './data/kold_v1_split_val.json'
test_paths = ['./data/kold_v1_split_test.json'] \
            + [f'./data/kold_v1_split_test_{r}.json' for r in ['10','30','50','70','90','natural_30','natural_50','natural_70']]
out_path = './data/words.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_paths, False)
with open(out_path, 'w', encoding='utf8')as f:
    f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))
