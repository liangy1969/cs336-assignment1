import json
from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH = "vocab.json"
MERGES_PATH = "merges.json"
SPECIAL_TOKEN = "<|endoftext|>"

def train():
    train_file_path = "E:/projects/cs336-assignment1/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = [
        SPECIAL_TOKEN
    ]
    vocab_size = 10000
    vocabs, merges = Tokenizer.train(train_file_path, vocab_size=vocab_size, special_tokens=special_tokens)

    # Save vocab: {hex_string: token_id}
    vocab_json = {v.hex(): k for k, v in vocabs.items()}
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)

    # Save merges: list of [hex1, hex2]
    merges_json = [[a.hex(), b.hex()] for a, b in merges]
    with open(MERGES_PATH, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2)


def load(
    vocab_path: str = VOCAB_PATH,
    merges_path: str = MERGES_PATH,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load vocab and merges from JSON files saved by train()."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {v: bytes.fromhex(k) for k, v in vocab_json.items()}

    with open(merges_path, "r", encoding="utf-8") as f:
        merges_json = json.load(f)
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_json]

    return vocab, merges


if __name__ == "__main__":
    train()

