from produce_emb import gen_embeddings_for_vocab
from train import args

def main():
    task = args.ex_task
    vocab_path = f"extrinsic/{task}/data/words.txt"
    #emb_path = f"extrinsic/{args.task}/data/love{oov}.emb"
    #emb_path = f"/mnt/data/oov/{task}/love_kor.emb"
    #ipa_path = None
    ipa_path = args.ex_ipa_path
    emb_path = args.ex_emb_path
    gen_embeddings_for_vocab(vocab_path=vocab_path, emb_path=emb_path, ipa_path=ipa_path)

if __name__ == '__main__':
    main()