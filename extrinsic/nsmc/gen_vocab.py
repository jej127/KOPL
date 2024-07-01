from utils import WordVocabulary

train_path = 'data/nsmc_train_cleaned.csv'
dev_path = 'data/nsmc_val_cleaned.csv'
test_paths = [f'data/nsmc_test_cleaned_{r}.csv' for r in [10,30,50,70,90]] + ['data/nsmc_test_cleaned.csv']
out_path = 'data/words_.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_paths, False)
with open(out_path, 'w', encoding='utf8')as f:
    f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))
