import json
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_tokenizer import load, SPECIAL_TOKEN

def tokenize():
    train_file_path = "E:/projects/cs336-assignment1/data/TinyStoriesV2-GPT4-train.txt"
    output_path = "E:/projects/cs336-assignment1/data/TinyStoriesV2-GPT4-train-tokens.json"
    # step 1: tokenize 
    vocab, merges = load()
    tokenizer = Tokenizer(vocab, merges, special_tokens=[SPECIAL_TOKEN])
    with open(train_file_path, "r", encoding="utf-8") as f:
        iterator = tokenizer.encode_iterable(f)
        train_token = [token for token in iterator]

    # step 2: save tokens to json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_token, f)
    

if __name__ == "__main__":
    tokenize()