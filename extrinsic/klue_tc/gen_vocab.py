from utils import WordVocabulary

train_path = './data/ynat-v1.1_train.json'
dev_path = './data/ynat-v1.1_dev.json'
test_paths = [f'./data/ynat-v1.1_dev_{r}.json' for r in [10,30,50,70,90]]
out_path = './data/words.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_paths, False)
with open(out_path, 'w', encoding='utf8')as f:
    f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))
