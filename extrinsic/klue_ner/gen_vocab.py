from utils import WordVocabulary

train_path = './data/train_split.txt'
dev_path = './data/dev_split.txt'
test_paths = ['./data/test_split.txt'] + [f'./data/test_split_{r}.txt' for r in [10,30,50,70,90]]
out_path = './data/words.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_paths, False)
with open(out_path, 'w', encoding='utf8')as f:
    f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))
