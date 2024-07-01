from utils import WordVocabulary

train_path = './data/klue-dp-v1.1_split_train.tsv'
dev_path = './data/klue-dp-v1.1_split_val.tsv'
test_paths = ['./data/klue-dp-v1.1_dev.tsv'] + [f'./data/klue-dp-v1.1_dev_{r}.tsv' for r in [10,30,50,70,90]]
out_path = './data/words.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_paths, False)
with open(out_path, 'w', encoding='utf8') as f:
    f.write('\n'.join([w.replace(' ','') for w in word_vocab._id_to_word]))
