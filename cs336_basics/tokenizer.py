import os
from typing import BinaryIO, Iterable, Iterator
import regex
import itertools


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PRE_TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # construct the reverse mapping
        self.vocab_bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_priority_dict = {}
        for priority, merge in enumerate(self.merges):
            self.merge_priority_dict[merge] = priority

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8", "replace")

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        pre_token_bytes = pre_token.encode("utf-8")
        token_ids = []
        pre_token_bytes_list = [bytes([b]) for b in pre_token_bytes]
        while True:
            # go through the pre_token_bytes_list until we find a merging pair
            best_merge_idx = None
            best_merge_priority = None
            for i in range(len(pre_token_bytes_list) - 1):
                bytes_pair = (pre_token_bytes_list[i], pre_token_bytes_list[i + 1])
                if bytes_pair in self.merge_priority_dict:
                    # merge the pair
                    merge_priority = self.merge_priority_dict[bytes_pair]
                    if best_merge_priority is None or merge_priority < best_merge_priority:
                        best_merge_priority = merge_priority
                        best_merge_idx = i
            if best_merge_idx is None:
                break
            pre_token_bytes_list_new = pre_token_bytes_list[:best_merge_idx] + [pre_token_bytes_list[best_merge_idx] + pre_token_bytes_list[best_merge_idx + 1]] + pre_token_bytes_list[best_merge_idx + 2:]
            pre_token_bytes_list = pre_token_bytes_list_new
        for pre_token_bytes in pre_token_bytes_list:
            token_ids.append(self.vocab_bytes_to_id[pre_token_bytes])
        return token_ids
    
    def _encode_split_text(self, split_text_list: list[str], special_tokens: list[str]) -> list[int]:
        token_ids = []
        # encode a list of text splitted by the special tokens
        for split_text in split_text_list:
            if split_text in special_tokens:
                token_ids.append(self.vocab_bytes_to_id[split_text.encode("utf-8")])
            else:
                # pre-tokenize the split_text using the same regex as in training
                for match in regex.finditer(PRE_TOKEN_REGEX, split_text):
                    pre_token = match.group(0)
                    token_ids.extend(self._encode_pre_token(pre_token))
        return token_ids


    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens
        if special_tokens is None:
            special_tokens = []
        # split the text by the special tokens; keep the special tokens in the split result
        if special_tokens:
            text_list = regex.split("(" + "|".join(map(regex.escape, special_tokens)) + ")", text)
        else:
            text_list = [text]
        return self._encode_split_text(text_list, special_tokens)
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_tokens = self.special_tokens
        if special_tokens is None:
            special_tokens = []

        it = iter(iterable)
        first = next(it, None)
            
        while first is not None:    
            if special_tokens:
                first_list = regex.split("(" + "|".join(map(regex.escape, special_tokens)) + ")", first)
            else:
                first_list = [first]
            # we want to check the last element of first_list
            if first_list[-1] in special_tokens:
                # last chunk ends on a special token — safe to encode everything and continue
                yield from self._encode_split_text(first_list, special_tokens)
                first = next(it, None)
            else:
                if len(first_list) > 1:
                    yield from self._encode_split_text(first_list[:-1], special_tokens)
                # the last element has no special token; may be a partial pre-token
                first = first_list[-1]
                pre_token_list = [match.group(0) for match in regex.finditer(PRE_TOKEN_REGEX, first)]
                if not pre_token_list:
                    # no pre-tokens at all; continue to next
                    first = next(it, None)
                    continue
                # encode all but the last pre-token
                for pre_token in pre_token_list[:-1]:
                    yield from self._encode_pre_token(pre_token)
                second = next(it, None)
                if second is None:
                    # no more text; encode the last pre-token as is
                    yield from self._encode_pre_token(pre_token_list[-1])
                    return 
                else:
                    # merge the last pre-token with the second text
                    first = pre_token_list[-1] + second

    @staticmethod
    def _pre_tokenize(corpus: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
        """ 
        Pre-tokenize the corpus using a regex, and count the frequency of each pre-token.
        Represent each pre-token as a tuple of base tokens (ints)
        """

        # split the special tokens before pre-tokenization, so that they are not merged with other tokens
        if special_tokens:
            corpus_list = regex.split("|".join(map(regex.escape, special_tokens)), corpus)
        else:
            corpus_list = [corpus]
        pre_token_counts: dict[tuple[int, ...], int] = {}
        for split_corpus in corpus_list:
            for match in regex.finditer(PRE_TOKEN_REGEX, split_corpus):
                pre_token_bytes = match.group(0).encode("utf-8")
                pre_token_bytes_tuple = tuple(pre_token_bytes)
                pre_token_counts[pre_token_bytes_tuple] = pre_token_counts.get(pre_token_bytes_tuple,  0) + 1

        return pre_token_counts
    
    @staticmethod
    def train(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train a BPE tokenizer on the input file, and return the vocab and merges.
        """
        with open(input_path, "rb") as f:
            corpus = f.read().decode("utf-8")
        return Tokenizer.train_from_corpus(corpus, vocab_size, special_tokens)
    
    
    @staticmethod
    def train_from_corpus(corpus: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train a BPE tokenizer on the input corpus, and return the vocab and merges.
        """
        pre_token_counts = Tokenizer._pre_tokenize(corpus, special_tokens)
        vocab = {i: bytes([i]) for i in range(256)}  # Start with base tokens
        merges = []

        # step 1: initialize the per pair counts based on the pre-token counts + pair to pre-token list mapping
        token_pair_counts: dict[tuple[int, int], int] = {}
        token_pair_to_pre_tokens: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for pre_token_tuple, count in pre_token_counts.items():
            for i in range(len(pre_token_tuple) - 1):
                token_pair = (pre_token_tuple[i], pre_token_tuple[i + 1])
                token_pair_counts[token_pair] = token_pair_counts.get(token_pair, 0) + count
                if token_pair not in token_pair_to_pre_tokens:
                    token_pair_to_pre_tokens[token_pair] = set()
                token_pair_to_pre_tokens[token_pair].add(pre_token_tuple)


        def iterative_merge() -> None:
            # given a pair counts dict, find the most common one and merge the token
            # update the pair count dict given the merged result:
            # we only need to update the pre-tokens that contain the merged pair
            # we go through the affected pre-tokens: 
            # 1. get all the pairs in the pre-token
            # 2. decrease the count of the old pairs
            # 3. reconstruct the pre-token with the merged token,
            # 4. increase the count of the new pairs
            most_common_pair = max(token_pair_counts, key=lambda p: (token_pair_counts[p], vocab[p[0]], vocab[p[1]]))
            merges.append((vocab[most_common_pair[0]], vocab[most_common_pair[1]]))
            new_token_id = max(vocab.keys()) + 1
            vocab[new_token_id] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
            affected_pre_tokens = token_pair_to_pre_tokens.pop(most_common_pair)
            token_pair_counts.pop(most_common_pair)
            for pre_token_tuple in affected_pre_tokens:
                pre_token_count = pre_token_counts.pop(pre_token_tuple)
                for i in range(len(pre_token_tuple)-1):
                    token_pair = (pre_token_tuple[i], pre_token_tuple[i+1])
                    if token_pair != most_common_pair:
                        token_pair_counts[token_pair] -= pre_token_count
                        if pre_token_tuple in token_pair_to_pre_tokens[token_pair]:
                            token_pair_to_pre_tokens[token_pair].remove(pre_token_tuple)
                # rewrite the pre_token_tuple with the merged token
                pre_token_new = []
                j = 0
                while j < len(pre_token_tuple):
                    if j < len(pre_token_tuple) - 1 and (pre_token_tuple[j], pre_token_tuple[j+1]) == most_common_pair:
                        # actual merging happens
                        pre_token_new.append(new_token_id)
                        # skip the next token since it's merged
                        j += 2
                    else:
                        pre_token_new.append(pre_token_tuple[j])
                        j += 1
                pre_token_tuple_new = tuple(pre_token_new)
                pre_token_counts[pre_token_tuple_new] = pre_token_count
                for i in range(len(pre_token_tuple_new)-1):
                    token_pair = (pre_token_tuple_new[i], pre_token_tuple_new[i + 1])
                    token_pair_counts[token_pair] = token_pair_counts.get(token_pair, 0) + pre_token_count
                    if token_pair not in token_pair_to_pre_tokens:
                        token_pair_to_pre_tokens[token_pair] = set()
                    token_pair_to_pre_tokens[token_pair].add(pre_token_tuple_new)

        while len(vocab) < vocab_size - len(special_tokens):
            iterative_merge()

        # final step: add the special tokens to the vocab
        for special_token in special_tokens:
            vocab[max(vocab.keys()) + 1] = special_token.encode("utf-8")

        return vocab, merges








